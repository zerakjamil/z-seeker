use anyhow::Result;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use tokio::sync::mpsc;
use tracing::{error, info};

pub struct FileWatcher {
    _watcher: RecommendedWatcher,
}

impl FileWatcher {
    pub fn new<P: AsRef<Path>>(watch_path: P, tx: mpsc::Sender<Event>) -> Result<Self> {
        let watch_path = watch_path.as_ref().to_path_buf();
        let mut _watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                match res {
                    Ok(event) => {
                        // In a real implementation we would debounce and filter using `ignore`
                        let _ = tx.blocking_send(event);
                    }
                    Err(e) => error!("Watch error: {:?}", e),
                }
            },
            Config::default(),
        )?;

        _watcher.watch(&watch_path, RecursiveMode::Recursive)?;
        info!("Started watching directory: {:?}", watch_path);

        Ok(Self { _watcher })
    }
}
