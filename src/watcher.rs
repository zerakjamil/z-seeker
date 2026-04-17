use anyhow::Result;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info};

const DEBOUNCE_WINDOW: Duration = Duration::from_millis(250);
const EVENT_CACHE_TTL: Duration = Duration::from_secs(5);

pub struct FileWatcher {
    _watcher: RecommendedWatcher,
}

#[derive(Clone)]
pub struct WatchConfig {
    pub max_file_size: u64,
    pub max_file_count: usize,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            max_file_size: 5242880, // 5MB
            max_file_count: 2000,
        }
    }
}

impl FileWatcher {
    pub fn new<P: AsRef<Path>>(
        watch_path: P,
        tx: mpsc::Sender<Event>,
        config: WatchConfig,
    ) -> Result<Self> {
        let watch_path = watch_path.as_ref().to_path_buf();
        let recent_events = Arc::new(Mutex::new(HashMap::<PathBuf, Instant>::new()));
        let recent_events_for_callback = Arc::clone(&recent_events);

        let mut _watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                match res {
                    Ok(event) => {
                        let now = Instant::now();
                        let should_send = {
                            let mut cache = recent_events_for_callback
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            should_forward_event(&event.paths, now, &mut cache, DEBOUNCE_WINDOW)
                        };

                        if should_send {
                            let _ = tx.blocking_send(event);
                        }
                    }
                    Err(e) => error!("Watch error: {:?}", e),
                }
            },
            Config::default(),
        )?;

        _watcher.watch(&watch_path, RecursiveMode::Recursive)?;
        info!("Started watching directory: {:?}", watch_path);
        info!(
            "Watch config: max_file_size={}, max_file_count={}",
            config.max_file_size, config.max_file_count
        );

        Ok(Self { _watcher })
    }
}

fn should_forward_event(
    paths: &[PathBuf],
    now: Instant,
    cache: &mut HashMap<PathBuf, Instant>,
    debounce_window: Duration,
) -> bool {
    cache.retain(|_, last_seen| now.duration_since(*last_seen) <= EVENT_CACHE_TTL);

    let mut should_send = false;

    for path in paths {
        let is_new_or_expired = match cache.get(path) {
            Some(last_seen) => now.duration_since(*last_seen) >= debounce_window,
            None => true,
        };

        if is_new_or_expired {
            should_send = true;
        }

        cache.insert(path.clone(), now);
    }

    should_send
}

#[cfg(test)]
mod tests {
    use super::should_forward_event;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[test]
    fn suppresses_duplicate_events_within_debounce_window() {
        let mut cache = HashMap::new();
        let path = PathBuf::from("src/main.rs");
        let start = Instant::now();
        let debounce_window = Duration::from_millis(250);

        assert!(should_forward_event(
            std::slice::from_ref(&path),
            start,
            &mut cache,
            debounce_window,
        ));
        assert!(!should_forward_event(
            std::slice::from_ref(&path),
            start + Duration::from_millis(100),
            &mut cache,
            debounce_window,
        ));
    }

    #[test]
    fn forwards_events_after_debounce_window() {
        let mut cache = HashMap::new();
        let path = PathBuf::from("src/main.rs");
        let start = Instant::now();
        let debounce_window = Duration::from_millis(250);

        assert!(should_forward_event(
            std::slice::from_ref(&path),
            start,
            &mut cache,
            debounce_window,
        ));
        assert!(should_forward_event(
            std::slice::from_ref(&path),
            start + Duration::from_millis(300),
            &mut cache,
            debounce_window,
        ));
    }
}
