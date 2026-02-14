// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    io,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex as StdMutex},
    time::{SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use external_storage::ExternalStorage;
use tikv_util::lru::{LruCache, SizePolicy};
use tokio::{
    io::AsyncWriteExt,
    sync::{Mutex as TokioMutex, Notify, mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::compat::FuturesAsyncReadCompatExt;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct CachedObject {
    entry: Arc<CacheFile>,
}

impl CachedObject {
    pub fn path(&self) -> &Path {
        &self.entry.path
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CacheAccessStat {
    pub cache_hit: u64,
    pub cache_miss: u64,
    pub cache_inflight_wait: u64,
    pub cache_evicted_files: u64,
    pub cache_evicted_bytes: u64,
    pub remote_read_calls: u64,
    pub remote_read_bytes: u64,
}

impl CacheAccessStat {
    fn hit() -> Self {
        Self {
            cache_hit: 1,
            ..Default::default()
        }
    }

    fn miss(remote_read_bytes: u64) -> Self {
        Self {
            cache_miss: 1,
            remote_read_calls: 1,
            remote_read_bytes,
            ..Default::default()
        }
    }

    fn inflight_wait() -> Self {
        Self {
            cache_inflight_wait: 1,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
struct CacheFile {
    key: String,
    path: PathBuf,
    size: u64,
    cleanup_tx: mpsc::UnboundedSender<CleanupMsg>,
}

impl Drop for CacheFile {
    fn drop(&mut self) {
        // Best-effort: cleanup is opportunistic and will be drained by the cache.
        let _ = self.cleanup_tx.send(CleanupMsg::Delete(CleanupRequest {
            key: self.key.clone(),
            path: self.path.clone(),
        }));
    }
}

#[derive(Debug, Clone)]
struct CleanupRequest {
    key: String,
    path: PathBuf,
}

#[derive(Debug)]
enum CleanupMsg {
    Delete(CleanupRequest),
    Shutdown(oneshot::Sender<()>),
}

#[derive(Default)]
struct ByteSizePolicy(usize);

impl SizePolicy<String, Arc<CacheFile>> for ByteSizePolicy {
    fn current(&self) -> usize {
        self.0
    }

    fn on_insert(&mut self, _key: &String, value: &Arc<CacheFile>) {
        self.0 = self
            .0
            .saturating_add(usize::try_from(value.size).unwrap_or(usize::MAX));
    }

    fn on_remove(&mut self, _key: &String, value: &Arc<CacheFile>) {
        self.0 = self
            .0
            .saturating_sub(usize::try_from(value.size).unwrap_or(usize::MAX));
    }

    fn on_reset(&mut self, val: usize) {
        self.0 = val;
    }
}

type CacheLru = LruCache<String, Arc<CacheFile>, ByteSizePolicy>;

fn new_cache_lru(capacity_bytes: usize) -> CacheLru {
    // Use LruCache's own eviction logic (EvictOnFull) driven by a byte-based
    // SizePolicy. Evicted entries are dropped immediately, and
    // `CacheEntry::drop` will enqueue file cleanup. The actual filesystem
    // removal is performed opportunistically.
    LruCache::with_capacity_sample_and_trace(capacity_bytes, 0, ByteSizePolicy::default())
}

#[derive(Debug, Clone)]
struct InFlightError {
    kind: io::ErrorKind,
    message: String,
}

impl From<InFlightError> for io::Error {
    fn from(value: InFlightError) -> Self {
        io::Error::new(value.kind, value.message)
    }
}

#[derive(Default)]
struct InFlight {
    done: Notify,
    result: StdMutex<Option<std::result::Result<Arc<CacheFile>, InFlightError>>>,
}

impl InFlight {
    fn is_finished(&self) -> bool {
        self.result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    async fn wait(&self) -> io::Result<Arc<CacheFile>> {
        loop {
            // Register the waiter first to avoid missing a `notify_waiters` between
            // checking `result` and awaiting the notification.
            let notified = self.done.notified();

            if let Some(res) = self
                .result
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
            {
                return res.map_err(Into::into);
            }
            notified.await;
        }
    }

    fn finish_ok(&self, entry: Arc<CacheFile>) {
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ok(entry));
        self.done.notify_waiters();
    }

    fn finish_err(&self, err: io::Error) {
        let inflight_err = InFlightError {
            kind: err.kind(),
            message: err.to_string(),
        };
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(Err(inflight_err));
        self.done.notify_waiters();
    }

    fn finish_cancelled(&self) {
        let mut guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_some() {
            return;
        }
        *guard = Some(Err(InFlightError {
            kind: io::ErrorKind::Interrupted,
            message: "inflight download cancelled".to_owned(),
        }));
        drop(guard);
        self.done.notify_waiters();
    }
}

struct InFlightGuard<'a> {
    key: String,
    inflight: Arc<InFlight>,
    map: &'a DashMap<String, Arc<InFlight>>,
}

impl<'a> InFlightGuard<'a> {
    fn new(key: String, inflight: Arc<InFlight>, map: &'a DashMap<String, Arc<InFlight>>) -> Self {
        Self { key, inflight, map }
    }
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        // If the in-flight is already finished, don't override the result. Still
        // do a best-effort cleanup of the map entry.
        if self.inflight.is_finished() {
            if let Some(existing) = self.map.get(&self.key) {
                if Arc::ptr_eq(existing.value(), &self.inflight) {
                    drop(existing);
                    self.map.remove(&self.key);
                }
            }
            return;
        }

        // If the "creator" future is cancelled/dropped, the in-flight entry would
        // otherwise stay in the map forever, and all waiters would wait forever.
        self.inflight.finish_cancelled();

        // Best-effort: only remove if it still points to the same in-flight instance.
        if let Some(existing) = self.map.get(&self.key) {
            if Arc::ptr_eq(existing.value(), &self.inflight) {
                drop(existing);
                self.map.remove(&self.key);
            }
        }
    }
}

/// A per-execution local cache for materializing remote log objects.
///
/// It provides:
/// - Per-object in-flight de-duplication for downloads.
/// - Global capacity control (best-effort LRU eviction, by whole-object files).
/// - Atomic file publishing (temp + rename).
pub struct LocalObjectCache {
    dir: PathBuf,
    capacity_bytes: usize,
    state: Arc<StdMutex<CacheLru>>,
    inflight: DashMap<String, Arc<InFlight>>,

    cleanup_tx: mpsc::UnboundedSender<CleanupMsg>,
    cleanup_task: TokioMutex<Option<JoinHandle<()>>>,
}

impl LocalObjectCache {
    pub async fn new(dir: PathBuf, capacity_bytes: u64) -> io::Result<Self> {
        let pid = std::process::id();
        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let dir = dir.join(format!("run_{}_{}", pid, ts_ms));
        tokio::fs::create_dir_all(&dir).await?;

        let capacity_bytes = usize::try_from(capacity_bytes).unwrap_or(usize::MAX);

        let (cleanup_tx, cleanup_rx) = mpsc::unbounded_channel();
        let state = Arc::new(StdMutex::new(new_cache_lru(capacity_bytes)));
        let cleanup_task = tokio::spawn(cleanup_worker(cleanup_rx, Arc::clone(&state)));

        Ok(Self {
            dir,
            capacity_bytes,
            state,
            inflight: DashMap::default(),

            cleanup_tx,
            cleanup_task: TokioMutex::new(Some(cleanup_task)),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn capacity_bytes(&self) -> u64 {
        u64::try_from(self.capacity_bytes).unwrap_or(u64::MAX)
    }

    pub async fn get_or_fetch(
        &self,
        storage: &dyn ExternalStorage,
        key: &str,
    ) -> io::Result<(CachedObject, CacheAccessStat)> {
        if let Some(entry) = self.get_hit(key).await? {
            return Ok((CachedObject { entry }, CacheAccessStat::hit()));
        }

        use dashmap::mapref::entry::Entry;
        let entry = self.inflight.entry(key.to_owned());
        match entry {
            Entry::Occupied(e) => {
                let inflight = Arc::clone(e.get());
                drop(e);
                let entry = inflight.wait().await?;
                Ok((CachedObject { entry }, CacheAccessStat::inflight_wait()))
            }
            Entry::Vacant(v) => {
                let inflight = Arc::new(InFlight::default());
                v.insert(Arc::clone(&inflight));

                let _guard =
                    InFlightGuard::new(key.to_owned(), Arc::clone(&inflight), &self.inflight);

                let res = self.fetch_and_insert(storage, key).await;
                match res {
                    Ok((entry, stat)) => {
                        inflight.finish_ok(Arc::clone(&entry));
                        self.inflight.remove(key);
                        Ok((CachedObject { entry }, stat))
                    }
                    Err(err) => {
                        inflight.finish_err(io::Error::new(err.kind(), err.to_string()));
                        self.inflight.remove(key);
                        Err(err)
                    }
                }
            }
        }
    }

    async fn get_hit(&self, key: &str) -> io::Result<Option<Arc<CacheFile>>> {
        let maybe_entry = {
            let mut lru = self.state.lock().unwrap_or_else(|e| e.into_inner());
            lru.get(&key.to_owned()).cloned()
        };
        let Some(entry) = maybe_entry else {
            return Ok(None);
        };

        match tokio::fs::metadata(&entry.path).await {
            Ok(meta) if meta.is_file() => Ok(Some(entry)),
            _ => {
                // The file is gone (or invalid). Drop the entry to keep the index consistent.
                // The dropped entry will enqueue a cleanup request, which will be a no-op if
                // the file is already missing.
                let removed = {
                    let mut lru = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    lru.remove(&key.to_owned())
                };
                drop(removed);
                Ok(None)
            }
        }
    }

    async fn fetch_and_insert(
        &self,
        storage: &dyn ExternalStorage,
        key: &str,
    ) -> io::Result<(Arc<CacheFile>, CacheAccessStat)> {
        let final_path = self.unique_path_for_key(key, Uuid::new_v4())?;
        if let Some(parent) = final_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let tmp_path = with_tmp_suffix(&final_path, Uuid::new_v4());
        let remote_read_bytes = match self.download_to(storage, key, &tmp_path).await {
            Ok(n) => n,
            Err(err) => {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Err(err);
            }
        };

        if let Err(err) = tokio::fs::rename(&tmp_path, &final_path).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(err);
        }

        let mut stat = CacheAccessStat::miss(remote_read_bytes);

        let entry = Arc::new(CacheFile {
            key: key.to_owned(),
            path: final_path.clone(),
            size: remote_read_bytes,
            cleanup_tx: self.cleanup_tx.clone(),
        });

        let (evicted, evicted_bytes) = self.insert_and_evict(key, Arc::clone(&entry)).await;
        stat.cache_evicted_files = evicted;
        stat.cache_evicted_bytes = evicted_bytes;
        Ok((entry, stat))
    }

    async fn download_to(
        &self,
        storage: &dyn ExternalStorage,
        key: &str,
        tmp_path: &Path,
    ) -> io::Result<u64> {
        let mut reader = storage.read(key).compat();
        let mut file = tokio::fs::File::create(tmp_path).await?;
        let n = tokio::io::copy(&mut reader, &mut file).await?;
        file.flush().await?;
        Ok(n)
    }

    async fn insert_and_evict(&self, key: &str, entry: Arc<CacheFile>) -> (u64, u64) {
        let (before_len, before_size, after_len, after_size) = {
            let mut lru = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let before_len = lru.len() as u64;
            let before_size = u64::try_from(lru.size()).unwrap_or(u64::MAX);
            let inserted_size = entry.size;

            lru.insert(key.to_owned(), entry);

            let after_len = lru.len() as u64;
            let after_size = u64::try_from(lru.size()).unwrap_or(u64::MAX);

            // NOTE: we return best-effort eviction stats based on LRU size changes.
            // Actual filesystem removals are performed by the background cleaner.
            (
                before_len,
                before_size.saturating_add(inserted_size),
                after_len,
                after_size,
            )
        };

        let evicted_files = before_len.saturating_add(1).saturating_sub(after_len);
        let evicted_bytes = before_size.saturating_sub(after_size);
        (evicted_files, evicted_bytes)
    }

    fn path_for_key(&self, key: &str) -> io::Result<PathBuf> {
        let rel = sanitize_relative_path(key)?;
        Ok(self.dir.join(rel))
    }

    fn unique_path_for_key(&self, key: &str, id: Uuid) -> io::Result<PathBuf> {
        let base = self.path_for_key(key)?;
        let Some(file_name) = base.file_name() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid external object name: {key}"),
            ));
        };
        let mut name = file_name.to_os_string();
        name.push(format!(".cache.{id}"));
        Ok(base.with_file_name(name))
    }

    pub async fn shutdown(&self) -> io::Result<()> {
        // Drop all entries, wait for background cleanup to drain, then remove the whole
        // run dir.
        {
            let mut lru = self.state.lock().unwrap_or_else(|e| e.into_inner());
            lru.clear();
        }

        if let Some(task) = self.cleanup_task.lock().await.take() {
            let (tx, rx) = oneshot::channel();
            let _ = self.cleanup_tx.send(CleanupMsg::Shutdown(tx));
            let _ = rx.await;
            let _ = task.await;
        }

        match tokio::fs::remove_dir_all(&self.dir).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }
}

async fn cleanup_worker(
    mut rx: mpsc::UnboundedReceiver<CleanupMsg>,
    state: Arc<StdMutex<CacheLru>>,
) {
    while let Some(msg) = rx.recv().await {
        match msg {
            CleanupMsg::Delete(item) => cleanup_one(item, &state).await,
            CleanupMsg::Shutdown(done) => {
                // Drain best-effort: process whatever is queued already, then exit.
                while let Ok(msg) = rx.try_recv() {
                    if let CleanupMsg::Delete(item) = msg {
                        cleanup_one(item, &state).await;
                    }
                }
                let _ = done.send(());
                break;
            }
        }
    }
}

async fn cleanup_one(item: CleanupRequest, state: &Arc<StdMutex<CacheLru>>) {
    let is_current = {
        let k = item.key.to_owned();
        let lru = state.lock().unwrap_or_else(|e| e.into_inner());
        lru
            .get_no_promote(&k)
            .is_some_and(|v| v.path == item.path)
    };
    if is_current {
        return;
    }

    if let Err(err) = tokio::fs::remove_file(&item.path).await {
        if err.kind() != io::ErrorKind::NotFound {
            tikv_util::warn!(
                "failed to remove cached file";
                "path" => %item.path.display(),
                "err" => %err
            );
        }
    }
}

fn with_tmp_suffix(path: &Path, id: Uuid) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp.{id}"));
    path.with_file_name(name)
}

fn sanitize_relative_path(key: &str) -> io::Result<PathBuf> {
    let p = Path::new(key);
    let mut rel = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(s) => rel.push(s),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) | Component::ParentDir => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid external object name: {key}"),
                ));
            }
        }
    }
    if rel.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid external object name: {key}"),
        ));
    }
    Ok(rel)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, Ordering},
        },
        time::Duration,
    };

    use async_trait::async_trait;
    use external_storage::{BlobObject, ExternalStorage};
    use futures::stream;
    use tokio::time::timeout;
    use tokio_util::compat::TokioAsyncReadCompatExt;

    use super::LocalObjectCache;

    struct BlockingStorage {
        data: Vec<u8>,
        block_next_read: AtomicBool,
        read_calls: AtomicU64,
        // Keep the writer end alive so the reader doesn't see EOF.
        _blocked_writer: std::sync::Mutex<Option<tokio::io::DuplexStream>>,
    }

    impl BlockingStorage {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                block_next_read: AtomicBool::new(true),
                read_calls: AtomicU64::new(0),
                _blocked_writer: std::sync::Mutex::new(None),
            }
        }

        fn read_calls(&self) -> u64 {
            self.read_calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl ExternalStorage for BlockingStorage {
        fn name(&self) -> &'static str {
            "blocking-storage"
        }

        fn url(&self) -> std::io::Result<url::Url> {
            Err(external_storage::unimplemented())
        }

        async fn write(
            &self,
            _name: &str,
            _reader: external_storage::UnpinReader<'_>,
            _content_length: u64,
        ) -> std::io::Result<()> {
            Err(external_storage::unimplemented())
        }

        fn read(&self, _name: &str) -> external_storage::ExternalData<'_> {
            self.read_calls.fetch_add(1, Ordering::Relaxed);

            if self.block_next_read.swap(false, Ordering::SeqCst) {
                let (r, w) = tokio::io::duplex(64);
                *self
                    ._blocked_writer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(w);
                Box::new(r.compat())
            } else {
                Box::new(futures::io::Cursor::new(self.data.clone()))
            }
        }

        fn read_part(
            &self,
            _name: &str,
            _off: u64,
            _len: u64,
        ) -> external_storage::ExternalData<'_> {
            Box::new(futures::io::Cursor::new(Vec::new()))
        }

        fn iter_prefix(
            &self,
            _prefix: &str,
        ) -> futures::stream::LocalBoxStream<'_, std::result::Result<BlobObject, std::io::Error>>
        {
            Box::pin(stream::empty())
        }

        fn delete(&self, _name: &str) -> futures::future::LocalBoxFuture<'_, std::io::Result<()>> {
            Box::pin(async { Err(external_storage::unimplemented()) })
        }
    }

    #[tokio::test]
    async fn test_inflight_creator_cancelled_wakes_waiters() {
        let tmp = tempdir::TempDir::new("compact-log-input-cache-cancel").unwrap();
        let cache = Arc::new(
            LocalObjectCache::new(tmp.path().to_path_buf(), 64 * 1024 * 1024)
                .await
                .unwrap(),
        );

        let storage = Arc::new(BlockingStorage::new(b"hello".to_vec()));
        let key = "obj";

        // Start a "creator" fetch that will block inside `read()`.
        let t1 = tokio::spawn({
            let storage = Arc::clone(&storage);
            let cache = Arc::clone(&cache);
            async move { cache.get_or_fetch(storage.as_ref(), key).await }
        });

        // Wait until the read() call has happened and inflight exists.
        timeout(Duration::from_secs(1), async {
            loop {
                if storage.read_calls() >= 1 && cache.inflight.get(key).is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let inflight = {
            let v = cache.inflight.get(key).unwrap();
            Arc::clone(v.value())
        };

        // A waiter should be released when the creator future is cancelled.
        let waiter = tokio::spawn(async move { inflight.wait().await });
        t1.abort();
        let _ = t1.await;

        let res = timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
        let err = res.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Interrupted);

        timeout(Duration::from_secs(1), async {
            loop {
                if cache.inflight.get(key).is_none() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        // A subsequent fetch should still work (the blocked creator was cleaned up).
        let (cached, _stat) = cache.get_or_fetch(storage.as_ref(), key).await.unwrap();
        let bytes = tokio::fs::read(cached.path()).await.unwrap();
        assert_eq!(bytes, b"hello");

        cache.shutdown().await.unwrap();
    }
}
