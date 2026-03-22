use anyhow::Result;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use tokio::sync::mpsc;
use tracing::{error, info};

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
        let mut _watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                match res {
                    Ok(event) => {
                        // In a real implementation we would debounce and filter using `ignore`
                        // Using max_file_size and max_file_count bounds in processing
                        let _ = tx.blocking_send(event);
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
