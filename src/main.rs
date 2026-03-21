mod auth;
mod db;
mod mcp;
mod parser;
mod provider;
mod search;
mod watcher;

use anyhow::Result;
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use std::env;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, Level};

use crate::db::VectorDb;
use crate::mcp::McpServer;
use crate::parser::CodeParser;
use crate::provider::github::CopilotProvider;
use crate::watcher::{FileWatcher, WatchConfig};

#[derive(Parser)]
#[command(name = "zseek", about = "Local Semantic Search MCP Server & CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Authenticate with GitHub Copilot
    Auth,
    /// Run the MCP server (Default if no command is provided)
    Mcp,
    /// Watch the current repository and keep the vector store in sync
    Watch {
        /// Limit uploads to files under a certain size (in bytes). Default: 5MB
        #[arg(long, default_value_t = 5242880)]
        max_file_size: u64,
        /// Maximum number of files to index. Default: 2000
        #[arg(long, default_value_t = 2000)]
        max_file_count: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env definitions
    dotenv().ok();

    // Setup logging (strictly stderr so it doesn't corrupt stdout MCP stream)
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Mcp) {
        Commands::Auth => {
            auth::run_auth_flow().await?;
        }
        Commands::Mcp => {
            info!("Starting Local Semantic Search MCP Server...");
            let (provider, db) = setup_core().await?;
            
            // Also start background watcher for MCP with defaults, or we can choose to rely on manual sync
            // We'll keep it as default for now
            let (tx, mut rx) = mpsc::channel(100);
            let current_dir = env::current_dir()?;
            let _watcher = FileWatcher::new(&current_dir, tx, WatchConfig::default())?;
            start_background_processor(provider.clone(), db.clone(), rx, WatchConfig::default());

            // Start MCP STDIO server
            let mcp_server = McpServer::new(provider, db);
            mcp_server.run().await?;
        }
        Commands::Watch { max_file_size, max_file_count } => {
            info!("Starting file watcher and indexing...");
            let (provider, db) = setup_core().await?;
            let current_dir = env::current_dir()?;

            let config = WatchConfig {
                max_file_size,
                max_file_count,
            };

            // Initiate the initial index (placeholder - we can add actual walkdir later)
            info!("Performing initial index of up to {} files, size limit {} bytes...", max_file_count, max_file_size);
            initial_index(&current_dir, &provider, &db, &config).await?;

            let (tx, mut rx) = mpsc::channel(100);
            let _watcher = FileWatcher::new(&current_dir, tx, config.clone())?;
            
            start_background_processor(provider.clone(), db.clone(), rx, config);

            // Block forever on the watcher
            info!("Watching for file changes. Press Ctrl+C to exit.");
            tokio::signal::ctrl_c().await?;
            info!("Shutting down watcher.");
        }
    }

    Ok(())
}

async fn setup_core() -> Result<(Arc<CopilotProvider>, Arc<VectorDb>)> {
    let token = env::var("COPILOT_API_KEY").unwrap_or_else(|_| {
        crate::auth::load_saved_token().unwrap_or_default()
    });

    if token.is_empty() {
        eprintln!("Authentication required! Run `zseek auth` in your terminal first.");
        std::process::exit(1);
    }
    
    let provider = Arc::new(CopilotProvider::new(token.clone()));

    let current_dir = env::current_dir()?;
    let db_path = current_dir.join(".lancedb");
    let lancedb_store = db_path.to_str().unwrap().to_string();
    let db = Arc::new(VectorDb::new(&lancedb_store).await?);

    Ok((provider, db))
}

fn start_background_processor(
    provider: Arc<CopilotProvider>,
    db: Arc<VectorDb>,
    mut rx: mpsc::Receiver<notify::Event>,
    config: WatchConfig,
) {
    tokio::spawn(async move {
        let mut parser = CodeParser::new();
        
        while let Some(event) = rx.recv().await {
            for path in event.paths {
                if !path.is_file() || path.starts_with(".git") || path.starts_with("target") {
                    continue;
                }
                
                // Enforce max file size
                if let Ok(metadata) = tokio::fs::metadata(&path).await {
                    if metadata.len() > config.max_file_size {
                        info!("Skipping large file: {} ({} bytes)", path.display(), metadata.len());
                        continue;
                    }
                }

                let content = match tokio::fs::read_to_string(&path).await {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                let chunks = match parser.parse_file(&path, &content) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Failed to parse {}: {}", path.display(), e);
                        continue;
                    }
                };
                
                if !chunks.is_empty() {
                    info!("File changed: {} ({} chunks extracted)", path.display(), chunks.len());
                }

                for chunk in chunks {
                    match provider.get_embeddings(&chunk.content).await {
                        Ok(embedding) => {
                            info!("Adding chunk from {} to VectorDb", chunk.file_path);
                            if let Err(e) = db.add_chunk(chunk, embedding).await {
                                error!("Failed to add chunk to db: {}", e);
                            }
                        },
                        Err(e) => {
                            error!("Copilot API Embedding error: {}", e);
                        }
                    }
                }
            }
        }
    });
}

async fn initial_index(
    dir: &std::path::Path,
    provider: &Arc<CopilotProvider>,
    db: &Arc<VectorDb>,
    config: &WatchConfig,
) -> Result<()> {
    // Implement an initial walking of the directory and indexing using `ignore`
    use ignore::WalkBuilder;
    let mut builder = WalkBuilder::new(dir);
    builder.hidden(true).git_ignore(true);
    
    let mut count = 0;
    let mut parser = CodeParser::new();

    for result in builder.build() {
        if count >= config.max_file_count {
            info!("Reached max_file_count of {}", config.max_file_count);
            break;
        }

        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        if metadata.len() > config.max_file_size {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let chunks = match parser.parse_file(path, &content) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if chunks.is_empty() {
            continue;
        }

        count += 1;
        info!("Indexing file {}/{}: {}", count, config.max_file_count, path.display());

        for chunk in chunks {
            if let Ok(embedding) = provider.get_embeddings(&chunk.content).await {
                let _ = db.add_chunk(chunk, embedding).await;
            }
        }
    }

    info!("Initial indexing complete. Indexed {} files.", count);
    Ok(())
}
