mod auth;
mod db;
mod install;
mod mcp;
mod parser;
mod provider;
mod search;
mod workspace;
mod watcher;

use anyhow::Result;
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{error, info, Level};

use crate::db::{SearchResult, VectorDb};
use crate::mcp::McpServer;
use crate::parser::CodeParser;
use crate::provider::github::CopilotProvider;
use crate::workspace::resolve_scope_roots;
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
    /// Install and register Z-Seeker to VS Code's settings automatically
    Install,
    /// Run the MCP server (Default if no command is provided)
    Mcp {
        /// Optional VS Code .code-workspace file; all listed folders become active roots
        #[arg(long)]
        workspace_file: Option<PathBuf>,
    },
    /// Watch the current repository and keep the vector store in sync
    Watch {
        /// Limit uploads to files under a certain size (in bytes). Default: 5MB
        #[arg(long, default_value_t = 5242880)]
        max_file_size: u64,
        /// Maximum number of files to index. Default: 2000
        #[arg(long, default_value_t = 2000)]
        max_file_count: usize,
        /// Optional VS Code .code-workspace file; all listed folders become active roots
        #[arg(long)]
        workspace_file: Option<PathBuf>,
    },
    /// Run retrieval quality benchmark against a query set
    Benchmark {
        /// Top-k retrieval limit for each query
        #[arg(long, default_value_t = 5)]
        limit: usize,
        /// Optional path to JSON benchmark cases (array of BenchmarkCase)
        #[arg(long)]
        cases_file: Option<PathBuf>,
        /// Path for saving/loading benchmark report history
        #[arg(long, default_value = ".zseek-benchmark-last.json")]
        report_file: PathBuf,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct BenchmarkCase {
    query: String,
    expected_file_contains: String,
    #[serde(default)]
    expected_symbol_contains: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkReport {
    generated_at_epoch_secs: u64,
    top_k: usize,
    total_cases: usize,
    hits: usize,
    no_result_cases: usize,
    total_results: usize,
    total_duplicates: usize,
    recall_at_k: f32,
    no_result_rate: f32,
    duplicate_rate: f32,
    avg_top_distance: Option<f32>,
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

    match cli.command.unwrap_or(Commands::Mcp {
        workspace_file: None,
    }) {
        Commands::Install => {
            if let Err(e) = install::install_to_vscode() {
                eprintln!("Failed to install to VS Code: {}", e);
            }
        }
        Commands::Auth => {
            auth::run_auth_flow().await?;
        }
        Commands::Mcp { workspace_file } => {
            info!("Starting Local Semantic Search MCP Server...");
            let current_dir = env::current_dir()?;
            let workspace_roots = resolve_scope_roots(&current_dir, workspace_file.as_deref())?;
            info!(
                "Active workspace roots ({}): {}",
                workspace_roots.len(),
                format_roots(&workspace_roots)
            );

            let (provider, db) = setup_core().await?;

            // Also start background watcher for MCP with defaults, or we can choose to rely on manual sync
            // We'll keep it as default for now
            let (tx, rx) = mpsc::channel(100);
            let mut _watchers = Vec::new();
            for root in &workspace_roots {
                _watchers.push(FileWatcher::new(root, tx.clone(), WatchConfig::default())?);
            }
            start_background_processor(provider.clone(), db.clone(), rx, WatchConfig::default());

            // Start MCP STDIO server
            let mcp_server = McpServer::new(provider, db, workspace_roots);
            mcp_server.run().await?;
        }
        Commands::Watch {
            max_file_size,
            max_file_count,
            workspace_file,
        } => {
            info!("Starting file watcher and indexing...");
            let current_dir = env::current_dir()?;
            let workspace_roots = resolve_scope_roots(&current_dir, workspace_file.as_deref())?;
            info!(
                "Active workspace roots ({}): {}",
                workspace_roots.len(),
                format_roots(&workspace_roots)
            );

            let (provider, db) = setup_core().await?;

            let config = WatchConfig {
                max_file_size,
                max_file_count,
            };

            // Initiate the initial index (placeholder - we can add actual walkdir later)
            info!(
                "Performing initial index of up to {} files per root, size limit {} bytes...",
                max_file_count,
                max_file_size
            );
            for root in &workspace_roots {
                info!("Initial indexing root: {}", root.display());
                initial_index(root, &provider, &db, &config).await?;
            }

            let (tx, rx) = mpsc::channel(100);
            let mut _watchers = Vec::new();
            for root in &workspace_roots {
                _watchers.push(FileWatcher::new(root, tx.clone(), config.clone())?);
            }

            start_background_processor(provider.clone(), db.clone(), rx, config);

            // Block forever on the watcher
            info!("Watching for file changes. Press Ctrl+C to exit.");
            tokio::signal::ctrl_c().await?;
            info!("Shutting down watcher.");
        }
        Commands::Benchmark {
            limit,
            cases_file,
            report_file,
        } => {
            info!("Running search quality benchmark...");
            let (provider, db) = setup_core().await?;
            run_benchmark(
                &provider,
                &db,
                limit,
                cases_file.as_deref(),
                report_file.as_path(),
            )
            .await?;
        }
    }

    Ok(())
}

fn format_roots(roots: &[PathBuf]) -> String {
    if roots.is_empty() {
        return "(none)".to_string();
    }

    roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

async fn setup_core() -> Result<(Arc<CopilotProvider>, Arc<VectorDb>)> {
    let token = env::var("COPILOT_API_KEY")
        .unwrap_or_else(|_| crate::auth::load_saved_token().unwrap_or_default());

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

fn is_ignored_path(path: &std::path::Path) -> bool {
    let ignored_components = [
        "node_modules",
        "vendor",
        ".git",
        "target",
        "dist",
        "build",
        "out",
        ".next",
        ".lancedb",
    ];

    for comp in path.components() {
        if let Some(comp_str) = comp.as_os_str().to_str() {
            if ignored_components.contains(&comp_str) {
                return true;
            }
        }
    }
    false
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
                if !path.is_file() || is_ignored_path(&path) {
                    continue;
                }

                // Enforce max file size
                if let Ok(metadata) = tokio::fs::metadata(&path).await {
                    if metadata.len() > config.max_file_size {
                        info!(
                            "Skipping large file: {} ({} bytes)",
                            path.display(),
                            metadata.len()
                        );
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
                        tracing::debug!("Failed to parse {}: {}", path.display(), e);
                        continue;
                    }
                };

                if !chunks.is_empty() {
                    info!(
                        "File changed: {} ({} chunks extracted)",
                        path.display(),
                        chunks.len()
                    );
                }

                let mut chunks_buffer = Vec::new();
                let mut seen_chunks = HashSet::new();

                for chunk in chunks {
                    let signature = chunk_signature(&chunk);
                    if !seen_chunks.insert(signature) {
                        continue;
                    }

                    match provider.get_embeddings(&chunk.content).await {
                        Ok(embedding) => {
                            chunks_buffer.push((chunk, embedding));
                        }
                        Err(e) => {
                            error!("Copilot API Embedding error: {}", e);
                        }
                    }
                }

                if !chunks_buffer.is_empty() {
                    info!("Adding {} chunks to VectorDb", chunks_buffer.len());
                    if let Err(e) = db.add_chunks(chunks_buffer).await {
                        error!("Failed to add chunks to db: {}", e);
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
    let mut batch_buffer = Vec::new();
    let mut seen_chunk_signatures = HashSet::new();

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
        if !path.is_file() || is_ignored_path(path) {
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
        info!(
            "Indexing file {}/{}: {}",
            count,
            config.max_file_count,
            path.display()
        );

        for chunk in chunks {
            let signature = chunk_signature(&chunk);
            if !seen_chunk_signatures.insert(signature) {
                continue;
            }

            if let Ok(embedding) = provider.get_embeddings(&chunk.content).await {
                batch_buffer.push((chunk, embedding));
            }
        }

        // Insert in batches of 200 chunks to avoid LanceDB version bloat
        if batch_buffer.len() >= 200 {
            info!(
                "Writing batch of {} chunks to vector database...",
                batch_buffer.len()
            );
            let batch = std::mem::take(&mut batch_buffer);
            if let Err(e) = db.add_chunks(batch).await {
                error!("Failed to add chunks: {}", e);
            }
        }
    }

    // Flush remaining
    if !batch_buffer.is_empty() {
        info!(
            "Writing final batch of {} chunks to vector database...",
            batch_buffer.len()
        );
        if let Err(e) = db.add_chunks(batch_buffer).await {
            error!("Failed to add chunks: {}", e);
        }
    }

    info!("Initial indexing complete. Indexed {} files.", count);
    Ok(())
}

fn chunk_signature(chunk: &crate::parser::Chunk) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    chunk.file_path.hash(&mut hasher);
    chunk.start_line.hash(&mut hasher);
    chunk.end_line.hash(&mut hasher);
    chunk.content.hash(&mut hasher);
    hasher.finish()
}

async fn run_benchmark(
    provider: &Arc<CopilotProvider>,
    db: &Arc<VectorDb>,
    limit: usize,
    cases_file: Option<&Path>,
    report_file: &Path,
) -> Result<()> {
    let cases = load_benchmark_cases(cases_file)?;
    if cases.is_empty() {
        println!("No benchmark cases found.");
        return Ok(());
    }

    let top_k = limit.clamp(1, 20);
    let mut hits = 0usize;
    let mut no_result_cases = 0usize;
    let mut total_duplicates = 0usize;
    let mut total_results = 0usize;
    let mut top_distance_sum = 0.0f32;
    let mut top_distance_count = 0usize;

    println!("Benchmarking {} queries with top-k={}...", cases.len(), top_k);

    for (idx, case) in cases.iter().enumerate() {
        let embedding = provider.get_embeddings(&case.query).await?;
        let results = db.search(embedding, top_k).await?;

        if results.is_empty() {
            no_result_cases += 1;
        }

        let is_hit = benchmark_hit(case, &results);
        if is_hit {
            hits += 1;
        }

        let duplicates = duplicate_count(&results);
        total_duplicates += duplicates;
        total_results += results.len();

        if let Some(distance) = results.first().and_then(|res| res.distance) {
            top_distance_sum += distance;
            top_distance_count += 1;
        }

        println!(
            "[{}] {} | hit={} | results={} | duplicates={}",
            idx + 1,
            case.query,
            is_hit,
            results.len(),
            duplicates
        );
    }

    let total_cases = cases.len() as f32;
    let recall_at_k = hits as f32 / total_cases;
    let no_result_rate = no_result_cases as f32 / total_cases;
    let duplicate_rate = if total_results == 0 {
        0.0
    } else {
        total_duplicates as f32 / total_results as f32
    };
    let avg_top_distance = if top_distance_count == 0 {
        None
    } else {
        Some(top_distance_sum / top_distance_count as f32)
    };

    println!("\n=== Benchmark Summary ===");
    println!("Recall@{}: {:.3}", top_k, recall_at_k);
    println!("No-result rate: {:.3}", no_result_rate);
    println!("Duplicate rate: {:.3}", duplicate_rate);
    if let Some(avg_distance) = avg_top_distance {
        println!("Average top-1 distance: {:.4}", avg_distance);
    } else {
        println!("Average top-1 distance: n/a");
    }

    let report = BenchmarkReport {
        generated_at_epoch_secs: current_epoch_secs(),
        top_k,
        total_cases: cases.len(),
        hits,
        no_result_cases,
        total_results,
        total_duplicates,
        recall_at_k,
        no_result_rate,
        duplicate_rate,
        avg_top_distance,
    };

    if let Some(previous) = load_previous_report(report_file)? {
        println!("\n=== Delta Vs Previous Run ===");
        print_metric_delta(
            &format!("Recall@{}", top_k),
            report.recall_at_k,
            previous.recall_at_k,
            false,
        );
        print_metric_delta(
            "No-result rate",
            report.no_result_rate,
            previous.no_result_rate,
            true,
        );
        print_metric_delta(
            "Duplicate rate",
            report.duplicate_rate,
            previous.duplicate_rate,
            true,
        );

        match (report.avg_top_distance, previous.avg_top_distance) {
            (Some(curr), Some(prev)) => {
                print_metric_delta("Average top-1 distance", curr, prev, true);
            }
            _ => println!("Average top-1 distance: n/a (insufficient data)"),
        }
    }

    save_benchmark_report(report_file, &report)?;
    println!("Saved benchmark report to {}", report_file.display());

    Ok(())
}

fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn load_previous_report(path: &Path) -> Result<Option<BenchmarkReport>> {
    if !path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(path)?;
    let report = serde_json::from_str::<BenchmarkReport>(&raw)?;
    Ok(Some(report))
}

fn save_benchmark_report(path: &Path, report: &BenchmarkReport) -> Result<()> {
    let raw = serde_json::to_string_pretty(report)?;
    std::fs::write(path, raw)?;
    Ok(())
}

fn signed_delta(delta: f32) -> String {
    if delta >= 0.0 {
        format!("+{:.3}", delta)
    } else {
        format!("{:.3}", delta)
    }
}

fn metric_trend(delta: f32, lower_is_better: bool) -> &'static str {
    if delta.abs() < 0.0005 {
        return "flat";
    }

    let improved = if lower_is_better {
        delta < 0.0
    } else {
        delta > 0.0
    };

    if improved {
        "improved"
    } else {
        "regressed"
    }
}

fn print_metric_delta(label: &str, current: f32, previous: f32, lower_is_better: bool) {
    let delta = current - previous;
    println!(
        "{}: {} ({})",
        label,
        signed_delta(delta),
        metric_trend(delta, lower_is_better)
    );
}

fn load_benchmark_cases(cases_file: Option<&Path>) -> Result<Vec<BenchmarkCase>> {
    if let Some(path) = cases_file {
        let raw = std::fs::read_to_string(path)?;
        let cases = serde_json::from_str::<Vec<BenchmarkCase>>(&raw)?;
        return Ok(cases);
    }

    Ok(default_benchmark_cases())
}

fn default_benchmark_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            query: "authentication token flow".to_string(),
            expected_file_contains: "src/auth.rs".to_string(),
            expected_symbol_contains: Some("run_auth_flow".to_string()),
        },
        BenchmarkCase {
            query: "vector database search query".to_string(),
            expected_file_contains: "src/db.rs".to_string(),
            expected_symbol_contains: Some("search".to_string()),
        },
        BenchmarkCase {
            query: "semantic search tool call handling".to_string(),
            expected_file_contains: "src/mcp.rs".to_string(),
            expected_symbol_contains: Some("rerank_chunks".to_string()),
        },
        BenchmarkCase {
            query: "code parser chunk extraction".to_string(),
            expected_file_contains: "src/parser.rs".to_string(),
            expected_symbol_contains: Some("parse_file".to_string()),
        },
        BenchmarkCase {
            query: "file watcher debounce behavior".to_string(),
            expected_file_contains: "src/watcher.rs".to_string(),
            expected_symbol_contains: Some("should_forward_event".to_string()),
        },
    ]
}

fn benchmark_hit(case: &BenchmarkCase, results: &[SearchResult]) -> bool {
    results.iter().any(|result| {
        let file_hit = result.chunk.file_path.contains(&case.expected_file_contains);
        let symbol_hit = case
            .expected_symbol_contains
            .as_deref()
            .and_then(|needle| {
                result
                    .chunk
                    .symbol_name
                    .as_deref()
                    .map(|name| name.contains(needle))
            })
            .unwrap_or(false);

        file_hit || symbol_hit
    })
}

fn duplicate_count(results: &[SearchResult]) -> usize {
    let mut seen = HashSet::new();
    let mut duplicates = 0usize;

    for result in results {
        let key = (
            result.chunk.file_path.clone(),
            result.chunk.start_line,
            result.chunk.end_line,
        );
        if !seen.insert(key) {
            duplicates += 1;
        }
    }

    duplicates
}

#[cfg(test)]
mod tests {
    use super::{benchmark_hit, duplicate_count, metric_trend, signed_delta, BenchmarkCase};
    use crate::db::SearchResult;
    use crate::parser::{content_hash_for_text, Chunk};

    fn sample_result(
        file_path: &str,
        start_line: usize,
        end_line: usize,
        symbol_name: Option<&str>,
    ) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                file_path: file_path.to_string(),
                content: "sample content".to_string(),
                start_line,
                end_line,
                language: "rust".to_string(),
                symbol_name: symbol_name.map(str::to_string),
                symbol_kind: Some("function".to_string()),
                content_hash: content_hash_for_text("sample content"),
            },
            distance: Some(0.2),
        }
    }

    #[test]
    fn benchmark_hit_matches_expected_file_or_symbol() {
        let case = BenchmarkCase {
            query: "auth flow".to_string(),
            expected_file_contains: "src/auth.rs".to_string(),
            expected_symbol_contains: Some("run_auth_flow".to_string()),
        };

        let results = vec![sample_result("src/auth.rs", 1, 5, None)];
        assert!(benchmark_hit(&case, &results));

        let symbol_results = vec![sample_result("src/other.rs", 1, 5, Some("run_auth_flow"))];
        assert!(benchmark_hit(&case, &symbol_results));
    }

    #[test]
    fn duplicate_count_counts_repeated_spans() {
        let results = vec![
            sample_result("src/auth.rs", 10, 20, Some("run_auth_flow")),
            sample_result("src/auth.rs", 10, 20, Some("run_auth_flow")),
            sample_result("src/auth.rs", 30, 40, Some("load_saved_token")),
        ];

        assert_eq!(duplicate_count(&results), 1);
    }

    #[test]
    fn signed_delta_formats_sign() {
        assert_eq!(signed_delta(0.1234), "+0.123");
        assert_eq!(signed_delta(-0.1234), "-0.123");
    }

    #[test]
    fn metric_trend_respects_directionality() {
        assert_eq!(metric_trend(0.02, false), "improved");
        assert_eq!(metric_trend(-0.02, false), "regressed");
        assert_eq!(metric_trend(-0.02, true), "improved");
        assert_eq!(metric_trend(0.02, true), "regressed");
        assert_eq!(metric_trend(0.0, true), "flat");
    }
}
