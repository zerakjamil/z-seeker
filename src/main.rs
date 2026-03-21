mod auth;
mod db;
mod mcp;
mod parser;
mod provider;
mod search;
mod watcher;

use anyhow::Result;
use dotenv::dotenv;
use std::env;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, Level};

use crate::db::VectorDb;
use crate::mcp::McpServer;
use crate::parser::CodeParser;
use crate::provider::github::CopilotProvider;
use crate::watcher::FileWatcher;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env definitions
    dotenv().ok();

    // Check for CLI args (e.g. `copilot-mcp-search auth`)
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "auth" {
        auth::run_auth_flow().await?;
        return Ok(());
    }

    // Setup logging (strictly stderr so it doesn't corrupt stdout MCP stream)
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_writer(std::io::stderr)
        .init();

    info!("Starting Local Semantic Search MCP Server...");

    // Determine config
    let current_dir = env::current_dir()?;
    let token = env::var("COPILOT_API_KEY").unwrap_or_else(|_| {
        crate::auth::load_saved_token().unwrap_or_default()
    });

    if token.is_empty() {
        eprintln!("Authentication required! Run `copilot-mcp-search auth` in your terminal first.");
        std::process::exit(1);
    }
    
    // Step 1: Initialize GitHub Copilot provider wrapped in Arc
    let provider = Arc::new(CopilotProvider::new(token.clone()));

    // Step 2: Initialize LanceDB storage wrapped in Arc
    let db_path = current_dir.join(".lancedb");
    let lancedb_store = db_path.to_str().unwrap().to_string();
    let db = Arc::new(VectorDb::new(&lancedb_store).await?);

    // Clone for background thread
    let bg_provider = provider.clone();
    let bg_db = db.clone();

    // Step 3: Start file watcher on background thread
    let (tx, mut rx) = mpsc::channel(100);
    let _watcher = match FileWatcher::new(&current_dir, tx) {
        Ok(w) => w,
        Err(e) => {
            error!("Background file watcher initialization failed: {}", e);
            return Err(e.into());
        }
    };

    // Step 4: Background orchestration loop for processing file changes
    tokio::spawn(async move {
        let mut parser = CodeParser::new();
        
        while let Some(event) = rx.recv().await {
            for path in event.paths {
                if !path.is_file() || path.starts_with(".git") || path.starts_with("target") {
                    continue;
                }
                
                let content = match tokio::fs::read_to_string(&path).await {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Pipeline: Chunk Code -> Embed -> Store
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
                    match bg_provider.get_embeddings(&chunk.content).await {
                        Ok(embedding) => {
                            info!("Adding chunk from {} to VectorDb", chunk.file_path);
                            if let Err(e) = bg_db.add_chunk(chunk, embedding).await {
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

    // Step 5: Start MCP STDIO server
    let mcp_server = McpServer::new(provider, db);
    mcp_server.run().await?;

    Ok(())
}
