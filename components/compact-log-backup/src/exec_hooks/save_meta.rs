use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.
use chrono::Local;
pub use engine_traits::SstCompressionType;
use external_storage::{ExternalStorage, UnpinReader};
use futures::{
    future::TryFutureExt,
    io::{AsyncReadExt, Cursor},
    stream::TryStreamExt,
};
use kvproto::brpb;
use protobuf::{Message, parse_from_bytes};
use tikv_util::{
    info,
    stream::{JustRetry, retry},
    warn,
};

use super::CollectStatistic;
use crate::{
    compaction::{META_OUT_REL, SST_OUT_REL, meta::CompactionRunInfoBuilder},
    errors::Result,
    execute::hooking::{
        AfterFinishCtx, BeforeStartCtx, CId, ExecHooks, SkipReason, SubcompactionFinishCtx,
        SubcompactionSkippedCtx, SubcompactionStartCtx,
    },
    statistic::CompactLogBackupStatistic,
};

/// Save the metadata to external storage after every subcompaction. After
/// everything done, it saves the whole compaction to a "migration" that can be
/// read by the BR CLI.
///
/// This is an essential plugin for real-world compacting, as single SST cannot
/// be restored.
///
/// "But why not just save the metadata of compaction in
/// [`SubcompactionExec`](crate::compaction::exec::SubcompactionExec)?"
///
/// First, As the hook system isn't exposed to end user, whether inlining this
/// is transparent to them -- they won't mistakely forget to add this hook and
/// ruin everything.
///
/// Also this make `SubcompactionExec` standalone, it will be easier to test.
///
/// The most important is, the hook knows metadata crossing subcompactions,
/// we can then optimize the arrangement of subcompactions (say, batching
/// subcompactoins), and save the final result in a single migration.
/// While [`SubcompactionExec`](crate::compaction::exec::SubcompactionExec)
/// knows only the subcompaction it handles, it is impossible to do such
/// optimizations.
pub struct SaveMeta {
    collector: CompactionRunInfoBuilder,
    stats: CollectStatistic,
    begin: chrono::DateTime<Local>,
    meta_writer: Option<MetaBatchWriter>,
}

impl Default for SaveMeta {
    fn default() -> Self {
        Self {
            collector: Default::default(),
            stats: Default::default(),
            begin: Local::now(),
            meta_writer: None,
        }
    }
}

/// A rolling batch writer for `.cmeta` objects.
///
/// It reduces object-count by writing batched `.cmeta` payloads, while still
/// persisting every finished subcompaction through immutable snapshots of the
/// current batch.
///
/// Each append writes a new `batch_{seq}_{version}.cmeta`, then deletes the
/// previous snapshot of the same batch only after the new snapshot is durable.
/// On resume we fall back to the latest valid snapshot, so a torn write can
/// lose at most the newest subcompaction of the active batch instead of the
/// entire batch.
struct MetaBatchWriter {
    dir: String,
    current_seq: u64,
    buffer: brpb::LogFileSubcompactions,
    current_snapshot_key: Option<String>,
    max_subcompactions_per_cmeta: usize,
    target_bytes_per_cmeta: usize,
}

#[derive(Debug, Clone)]
struct BatchSnapshot {
    seq: u64,
    version: u64,
    key: String,
}

impl BatchSnapshot {
    fn from_key(key: &str) -> Option<Self> {
        let file_name = Path::new(key).file_name()?.to_str()?;
        let body = file_name.strip_prefix("batch_")?.strip_suffix(".cmeta")?;
        let (seq, version) = match body.split_once('_') {
            Some((seq, version)) => (seq.parse().ok()?, version.parse().ok()?),
            None => (body.parse().ok()?, 0),
        };
        Some(Self {
            seq,
            version,
            key: key.to_owned(),
        })
    }

    fn new(dir: &str, seq: u64, version: u64) -> Self {
        Self {
            seq,
            version,
            key: format!("{}/batch_{:06}_{:06}.cmeta", dir, seq, version),
        }
    }
}

impl MetaBatchWriter {
    const DEFAULT_MAX_SUBCOMPACTIONS_PER_CMETA: usize = 128;
    const DEFAULT_TARGET_BYTES_PER_CMETA: usize = 4 * 1024 * 1024;

    fn new(
        dir: String,
        current_seq: u64,
        buffer: brpb::LogFileSubcompactions,
        current_snapshot_key: Option<String>,
    ) -> Self {
        Self {
            dir,
            current_seq,
            buffer,
            current_snapshot_key,
            max_subcompactions_per_cmeta: Self::DEFAULT_MAX_SUBCOMPACTIONS_PER_CMETA,
            target_bytes_per_cmeta: Self::DEFAULT_TARGET_BYTES_PER_CMETA,
        }
    }

    fn should_rotate(&self, current_bytes: usize) -> bool {
        self.buffer.subcompactions.len() >= self.max_subcompactions_per_cmeta
            || current_bytes >= self.target_bytes_per_cmeta
    }

    async fn read_snapshot(
        storage: &dyn ExternalStorage,
        key: &str,
    ) -> Result<brpb::LogFileSubcompactions> {
        let mut content = vec![];
        storage.read(key).read_to_end(&mut content).await?;
        Ok(parse_from_bytes(&content)?)
    }

    async fn write_snapshot(storage: &dyn ExternalStorage, key: &str, bytes: &[u8]) -> Result<()> {
        retry(|| async {
            let reader = UnpinReader(Box::new(Cursor::new(bytes)));
            storage
                .write(key, reader, bytes.len() as _)
                .map_err(JustRetry)
                .await
        })
        .await
        .map_err(|err| err.0)?;
        Ok(())
    }

    async fn delete_snapshot(storage: &dyn ExternalStorage, key: &str) -> Result<()> {
        retry(|| async { storage.delete(key).map_err(JustRetry).await })
            .await
            .map_err(|err| err.0)?;
        Ok(())
    }

    async fn cleanup_snapshots(
        storage: &dyn ExternalStorage,
        snapshots: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        for key in snapshots {
            Self::delete_snapshot(storage, &key).await?;
        }
        Ok(())
    }

    async fn load_or_new(storage: &dyn ExternalStorage, out_prefix: &str) -> Result<Self> {
        let dir = format!("{}/{}", out_prefix, META_OUT_REL);
        let list_prefix = format!("{}/", dir);

        let mut snapshots_by_seq = BTreeMap::<u64, Vec<BatchSnapshot>>::new();
        let mut stream = storage.iter_prefix(&list_prefix);
        while let Some(item) = stream.try_next().await? {
            let Some(snapshot) = BatchSnapshot::from_key(&item.key) else {
                continue;
            };
            snapshots_by_seq
                .entry(snapshot.seq)
                .or_default()
                .push(snapshot);
        }

        let Some((max_seq, mut snapshots)) = snapshots_by_seq.into_iter().next_back() else {
            return Ok(Self::new(dir, 0, brpb::LogFileSubcompactions::new(), None));
        };
        snapshots.sort_by(|lhs, rhs| rhs.version.cmp(&lhs.version));

        let mut chosen = None;
        for (idx, snapshot) in snapshots.iter().enumerate() {
            match Self::read_snapshot(storage, &snapshot.key).await {
                Ok(buffer) => {
                    if snapshot.version > 0
                        && buffer.subcompactions.len() as u64 != snapshot.version
                    {
                        warn!(
                            "SaveMeta: cmeta batch snapshot size mismatch, ignoring it.";
                            "key" => %snapshot.key,
                            "expected_subcompactions" => snapshot.version,
                            "actual_subcompactions" => buffer.subcompactions.len()
                        );
                        continue;
                    }
                    chosen = Some((idx, buffer));
                    break;
                }
                Err(err) => {
                    warn!(
                        "SaveMeta: failed to parse existing cmeta batch, ignoring it.";
                        "key" => %snapshot.key,
                        "err" => %err
                    );
                }
            }
        }

        match chosen {
            Some((chosen_idx, buffer)) => {
                let chosen_snapshot = snapshots[chosen_idx].clone();
                let stale = snapshots
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != chosen_idx)
                    .map(|(_, snapshot)| snapshot.key.clone())
                    .collect::<Vec<_>>();
                Self::cleanup_snapshots(storage, stale).await?;

                let mut this = Self::new(dir, max_seq, buffer, Some(chosen_snapshot.key.clone()));
                let current_bytes = this.buffer.write_to_bytes()?.len();
                if this.should_rotate(current_bytes) {
                    this.current_seq = max_seq + 1;
                    this.buffer = brpb::LogFileSubcompactions::new();
                    this.current_snapshot_key = None;
                }
                Ok(this)
            }
            None => {
                warn!(
                    "SaveMeta: no valid cmeta batch snapshot found, starting a new batch.";
                    "seq" => max_seq
                );
                Self::cleanup_snapshots(
                    storage,
                    snapshots.into_iter().map(|snapshot| snapshot.key),
                )
                .await?;
                Ok(Self::new(
                    dir,
                    max_seq + 1,
                    brpb::LogFileSubcompactions::new(),
                    None,
                ))
            }
        }
    }

    async fn append_and_flush(
        &mut self,
        storage: &dyn ExternalStorage,
        subcompaction: brpb::LogFileSubcompaction,
    ) -> Result<()> {
        self.buffer.mut_subcompactions().push(subcompaction);
        let bytes = self.buffer.write_to_bytes()?;
        let snapshot = BatchSnapshot::new(
            &self.dir,
            self.current_seq,
            self.buffer.subcompactions.len() as u64,
        );
        Self::write_snapshot(storage, &snapshot.key, &bytes).await?;

        if let Some(prev_key) = self.current_snapshot_key.replace(snapshot.key.clone()) {
            Self::delete_snapshot(storage, &prev_key).await?;
        }

        if self.should_rotate(bytes.len()) {
            self.current_seq += 1;
            self.buffer = brpb::LogFileSubcompactions::new();
            self.current_snapshot_key = None;
        }
        Ok(())
    }
}

impl SaveMeta {
    fn comments(&self) -> String {
        let now = Local::now();
        let stat = CompactLogBackupStatistic {
            start_time: self.begin,
            end_time: Local::now(),
            time_taken: (now - self.begin).to_std().unwrap_or_default(),
            exec_by: tikv_util::sys::hostname().unwrap_or_default(),

            load_stat: self.stats.load_stat.clone(),
            subcompact_stat: self.stats.compact_stat.clone(),
            load_meta_stat: self.stats.load_meta_stat.clone(),
            collect_subcompactions_stat: self.stats.collect_stat.clone(),
            prometheus: Default::default(),
        };
        serde_json::to_string(&stat).unwrap_or_else(|err| format!("ERR DURING MARSHALING: {}", err))
    }
}

impl ExecHooks for SaveMeta {
    async fn before_execution_started(&mut self, cx: BeforeStartCtx<'_>) -> Result<()> {
        self.begin = Local::now();
        self.meta_writer =
            Some(MetaBatchWriter::load_or_new(cx.storage, &cx.this.out_prefix).await?);
        let run_info = &mut self.collector;
        run_info.mut_meta().set_name(cx.this.gen_name());
        run_info
            .mut_meta()
            .set_compaction_from_ts(cx.this.cfg.from_ts);
        run_info
            .mut_meta()
            .set_compaction_until_ts(cx.this.cfg.until_ts);
        run_info
            .mut_meta()
            .set_artifacts(format!("{}/{}", cx.this.out_prefix, META_OUT_REL));
        run_info
            .mut_meta()
            .set_generated_files(format!("{}/{}", cx.this.out_prefix, SST_OUT_REL));
        Ok(())
    }

    fn before_a_subcompaction_start(&mut self, _cid: CId, c: SubcompactionStartCtx<'_>) {
        self.stats
            .update_collect_compaction_stat(c.collect_compaction_stat_diff);
        self.stats.update_load_meta_stat(c.load_stat_diff);
    }

    async fn on_subcompaction_skipped(&mut self, cx: SubcompactionSkippedCtx<'_>) {
        if cx.reason == SkipReason::AlreadyDone {
            self.collector.add_origin_subcompaction(cx.subc);
        }
    }

    async fn after_a_subcompaction_end(
        &mut self,
        _cid: CId,
        cx: SubcompactionFinishCtx<'_>,
    ) -> Result<()> {
        self.collector.add_subcompaction(cx.result);
        self.stats.update_subcompaction(cx.result);

        let Some(writer) = self.meta_writer.as_mut() else {
            return Err(crate::ErrorKind::Other(
                "SaveMeta: meta writer not initialized".to_owned(),
            )
            .into());
        };
        writer
            .append_and_flush(cx.external_storage, cx.result.meta.clone())
            .await?;
        Result::Ok(())
    }

    async fn after_execution_finished(&mut self, cx: AfterFinishCtx<'_>) -> Result<()> {
        if self.collector.is_empty() {
            warn!("Nothing to write, skipping saving meta.");
            return Ok(());
        }
        let comments = self.comments();
        self.collector.mut_meta().set_comments(comments);
        let begin = Instant::now();
        self.collector
            .write_migration(Arc::clone(cx.storage))
            .await?;
        info!("Migration written."; "duration" => ?begin.elapsed());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use external_storage::ExternalStorage;
    use futures::{io::Cursor, stream::TryStreamExt};
    use kvproto::brpb;
    use protobuf::Message;

    use super::MetaBatchWriter;
    use crate::{compaction::META_OUT_REL, test_util::TmpStorage};

    fn sample_subcompaction(region_id: u64) -> brpb::LogFileSubcompaction {
        let mut meta = brpb::LogFileSubcompactionMeta::new();
        meta.set_region_id(region_id);
        meta.set_cf("default".to_owned());
        meta.set_ty(brpb::FileType::Put);
        meta.set_size(region_id);
        meta.set_input_min_ts(region_id);
        meta.set_input_max_ts(region_id + 10);
        meta.set_compact_from_ts(1);
        meta.set_compact_until_ts(100);

        let mut subc = brpb::LogFileSubcompaction::new();
        subc.set_meta(meta);
        subc
    }

    async fn write_batch(
        storage: &dyn ExternalStorage,
        key: &str,
        subcompactions: Vec<brpb::LogFileSubcompaction>,
    ) {
        let mut batch = brpb::LogFileSubcompactions::new();
        batch.set_subcompactions(subcompactions.into());
        let bytes = batch.write_to_bytes().unwrap();
        storage
            .write(key, Cursor::new(bytes.clone()).into(), bytes.len() as u64)
            .await
            .unwrap();
    }

    async fn list_cmeta_keys(storage: &dyn ExternalStorage, prefix: &str) -> Vec<String> {
        let mut keys = storage
            .iter_prefix(prefix)
            .map_ok(|item| item.key)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        keys.sort();
        keys
    }

    #[tokio::test]
    async fn test_meta_batch_writer_keeps_only_latest_snapshot() {
        let st = TmpStorage::create();
        let dir = format!("test-output/{}", META_OUT_REL);
        let mut writer =
            MetaBatchWriter::new(dir.clone(), 0, brpb::LogFileSubcompactions::new(), None);

        writer
            .append_and_flush(st.storage().as_ref(), sample_subcompaction(1))
            .await
            .unwrap();
        assert_eq!(
            list_cmeta_keys(st.storage().as_ref(), &dir).await,
            vec![format!("{}/batch_000000_000001.cmeta", dir)]
        );

        writer
            .append_and_flush(st.storage().as_ref(), sample_subcompaction(2))
            .await
            .unwrap();
        assert_eq!(
            list_cmeta_keys(st.storage().as_ref(), &dir).await,
            vec![format!("{}/batch_000000_000002.cmeta", dir)]
        );

        let subcs = st.load_subcompactions(&dir).await.unwrap();
        assert_eq!(subcs.len(), 2);
        assert_eq!(subcs[0].get_meta().get_region_id(), 1);
        assert_eq!(subcs[1].get_meta().get_region_id(), 2);
    }

    #[tokio::test]
    async fn test_meta_batch_writer_loads_previous_valid_snapshot() {
        let st = TmpStorage::create();
        let out_prefix = "test-output";
        let dir = format!("{}/{}", out_prefix, META_OUT_REL);
        let valid_key = format!("{}/batch_000000_000001.cmeta", dir);
        let corrupt_key = format!("{}/batch_000000_000002.cmeta", dir);

        write_batch(
            st.storage().as_ref(),
            &valid_key,
            vec![sample_subcompaction(7)],
        )
        .await;
        st.storage()
            .write(
                &corrupt_key,
                Cursor::new(b"definitely-not-a-protobuf".to_vec()).into(),
                25,
            )
            .await
            .unwrap();

        let writer = MetaBatchWriter::load_or_new(st.storage().as_ref(), out_prefix)
            .await
            .unwrap();

        assert_eq!(writer.current_seq, 0);
        assert_eq!(writer.buffer.subcompactions.len(), 1);
        assert_eq!(
            writer.buffer.subcompactions[0].get_meta().get_region_id(),
            7
        );
        assert_eq!(
            writer.current_snapshot_key.as_deref(),
            Some(valid_key.as_str())
        );
        assert_eq!(
            list_cmeta_keys(st.storage().as_ref(), &dir).await,
            vec![valid_key]
        );
    }
}
