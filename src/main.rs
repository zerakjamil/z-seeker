mod auth;
mod db;
mod install;
mod mcp;
mod parser;
mod provider;
mod search;
mod workspace;
mod watcher;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{error, info, warn, Level};

use crate::db::{SearchResult, VectorDb};
use crate::mcp::McpServer;
use crate::parser::CodeParser;
use crate::provider::github::CopilotProvider;
use crate::workspace::resolve_scope_roots;
use crate::watcher::{FileWatcher, WatchConfig};

#[derive(Parser)]
#[command(name = "zseek", about = "Local Semantic Search MCP Server & CLI")]
struct Cli {
    /// Print resolved folders from a VS Code .code-workspace file and exit
    #[arg(long)]
    workspace_file_folders: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Authenticate with GitHub Copilot
    Auth,
    /// Install and register Z-Seeker to VS Code's settings automatically
    Install,
    /// Self-update zseek to the newest version from GitHub
    #[command(name = "self-update", visible_alias = "selfupdate")]
    SelfUpdate,
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
        /// Built-in benchmark dataset pack (ignored when --cases-file is provided)
        #[arg(long, default_value = "core")]
        dataset_pack: String,
        /// Enforce quality gates and fail benchmark command on violations
        #[arg(long, default_value_t = false)]
        enforce_gates: bool,
        /// Require non-regression against previous report metrics
        #[arg(long, default_value_t = false)]
        require_non_regression: bool,
        /// Allowed regression tolerance when non-regression checks are enabled
        #[arg(long, default_value_t = 0.0)]
        regression_tolerance: f32,
        /// Absolute minimum Recall@k when gates are enabled
        #[arg(long)]
        min_recall: Option<f32>,
        /// Absolute minimum Precision@k when gates are enabled
        #[arg(long)]
        min_precision: Option<f32>,
        /// Absolute minimum MRR@k when gates are enabled
        #[arg(long)]
        min_mrr: Option<f32>,
        /// Absolute minimum NDCG@k when gates are enabled
        #[arg(long)]
        min_ndcg: Option<f32>,
        /// Absolute maximum no-result rate when gates are enabled
        #[arg(long)]
        max_no_result_rate: Option<f32>,
        /// Absolute maximum duplicate rate when gates are enabled
        #[arg(long)]
        max_duplicate_rate: Option<f32>,
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
    #[serde(default)]
    task_family: Option<String>,
}

#[derive(Debug, Clone)]
struct BenchmarkGateConfig {
    enabled: bool,
    require_non_regression: bool,
    regression_tolerance: f32,
    min_recall_at_k: Option<f32>,
    min_precision_at_k: Option<f32>,
    min_mrr_at_k: Option<f32>,
    min_ndcg_at_k: Option<f32>,
    max_no_result_rate: Option<f32>,
    max_duplicate_rate: Option<f32>,
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
    #[serde(default)]
    precision_at_k: f32,
    #[serde(default)]
    mrr_at_k: f32,
    #[serde(default)]
    ndcg_at_k: f32,
    no_result_rate: f32,
    duplicate_rate: f32,
    avg_top_distance: Option<f32>,
}

struct WorkspaceIgnoreMatcher {
    root: PathBuf,
    matcher: Gitignore,
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

    if let Some(workspace_file) = cli.workspace_file_folders.as_deref() {
        let current_dir = env::current_dir()?;
        print_workspace_file_folders_preview(&current_dir, workspace_file)?;
        return Ok(());
    }

    match cli.command.unwrap_or(Commands::Mcp {
        workspace_file: None,
    }) {
        Commands::Install => {
            if let Err(e) = install::install_to_vscode() {
                eprintln!("Failed to install to VS Code: {}", e);
            }
        }
        Commands::SelfUpdate => {
            install::self_update()?;
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
            log_workspace_roots_for_indexing(&workspace_roots);
            ensure_workspace_git_local_excludes(&workspace_roots);
            let ignore_matchers = Arc::new(load_custom_ignore_matchers(&workspace_roots));

            let (provider, db) = setup_core().await?;
            let config = WatchConfig::default();

            let startup_roots = workspace_roots.clone();
            let startup_provider = provider.clone();
            let startup_db = db.clone();
            let startup_config = config.clone();
            let startup_ignore_matchers = ignore_matchers.clone();
            tokio::spawn(async move {
                info!(
                    "Starting background initial indexing for MCP (up to {} files per root, size limit {} bytes)...",
                    startup_config.max_file_count,
                    startup_config.max_file_size
                );
                for root in &startup_roots {
                    info!("Initial indexing root: {}", root.display());
                    if let Err(err) =
                        initial_index(
                            root,
                            &startup_provider,
                            &startup_db,
                            &startup_config,
                            startup_ignore_matchers.as_ref(),
                        )
                        .await
                    {
                        error!(
                            "Initial indexing failed for root {}: {}",
                            root.display(),
                            err
                        );
                    }
                }
                info!("Background initial indexing for MCP completed.");
            });

            // Also start background watcher for MCP with defaults, or we can choose to rely on manual sync
            // We'll keep it as default for now
            let (tx, rx) = mpsc::channel(100);
            let mut _watchers = Vec::new();
            for root in &workspace_roots {
                _watchers.push(FileWatcher::new(root, tx.clone(), config.clone())?);
            }
            start_background_processor(
                provider.clone(),
                db.clone(),
                rx,
                config,
                ignore_matchers,
            );

            // Start MCP STDIO server
            info!("MCP server ready; accepting requests while indexing continues in background.");
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
            log_workspace_roots_for_indexing(&workspace_roots);
            ensure_workspace_git_local_excludes(&workspace_roots);
            let ignore_matchers = Arc::new(load_custom_ignore_matchers(&workspace_roots));

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
                initial_index(root, &provider, &db, &config, ignore_matchers.as_ref()).await?;
            }

            let (tx, rx) = mpsc::channel(100);
            let mut _watchers = Vec::new();
            for root in &workspace_roots {
                _watchers.push(FileWatcher::new(root, tx.clone(), config.clone())?);
            }

            start_background_processor(
                provider.clone(),
                db.clone(),
                rx,
                config,
                ignore_matchers,
            );

            // Block forever on the watcher
            info!("Watching for file changes. Press Ctrl+C to exit.");
            tokio::signal::ctrl_c().await?;
            info!("Shutting down watcher.");
        }
        Commands::Benchmark {
            limit,
            cases_file,
            dataset_pack,
            enforce_gates,
            require_non_regression,
            regression_tolerance,
            min_recall,
            min_precision,
            min_mrr,
            min_ndcg,
            max_no_result_rate,
            max_duplicate_rate,
            report_file,
        } => {
            info!("Running search quality benchmark...");
            let (provider, db) = setup_core().await?;
            let gate_config = BenchmarkGateConfig {
                enabled: enforce_gates,
                require_non_regression,
                regression_tolerance,
                min_recall_at_k: min_recall,
                min_precision_at_k: min_precision,
                min_mrr_at_k: min_mrr,
                min_ndcg_at_k: min_ndcg,
                max_no_result_rate,
                max_duplicate_rate,
            };
            run_benchmark(
                &provider,
                &db,
                limit,
                cases_file.as_deref(),
                Some(dataset_pack.as_str()),
                &gate_config,
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

fn format_workspace_roots_plan(roots: &[PathBuf]) -> String {
    if roots.is_empty() {
        return "(none)".to_string();
    }

    roots
        .iter()
        .enumerate()
        .map(|(index, root)| format!("{}. {}", index + 1, root.display()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn print_workspace_file_folders_preview(current_dir: &Path, workspace_file: &Path) -> Result<()> {
    let workspace_roots = resolve_scope_roots(current_dir, Some(workspace_file))?;
    println!(
        "Workspace folders zseek will work on ({}):",
        workspace_roots.len()
    );
    println!("{}", format_workspace_roots_plan(&workspace_roots));
    Ok(())
}

fn log_workspace_roots_for_indexing(workspace_roots: &[PathBuf]) {
    info!("Folders selected for indexing:");
    for (index, root) in workspace_roots.iter().enumerate() {
        info!("  {}. {}", index + 1, root.display());
    }
}

fn ensure_workspace_git_local_excludes(workspace_roots: &[PathBuf]) {
    for root in workspace_roots {
        if let Err(err) = ensure_local_git_exclude_for_workspace(root) {
            warn!(
                "Failed to update local git excludes for {}: {}",
                root.display(),
                err
            );
        }
    }
}

fn resolve_git_dir_for_workspace(workspace_root: &Path) -> Option<PathBuf> {
    let git_entry = workspace_root.join(".git");

    if git_entry.is_dir() {
        return Some(git_entry);
    }

    if !git_entry.is_file() {
        return None;
    }

    let raw = std::fs::read_to_string(&git_entry).ok()?;
    let first_line = raw.lines().next()?.trim();
    let relative_git_dir = first_line.strip_prefix("gitdir:")?.trim();
    if relative_git_dir.is_empty() {
        return None;
    }

    let git_dir = PathBuf::from(relative_git_dir);
    Some(if git_dir.is_absolute() {
        git_dir
    } else {
        workspace_root.join(git_dir)
    })
}

fn ensure_local_git_exclude_for_workspace(workspace_root: &Path) -> Result<()> {
    let Some(git_dir) = resolve_git_dir_for_workspace(workspace_root) else {
        return Ok(());
    };

    let exclude_path = git_dir.join("info").join("exclude");
    if let Some(parent) = exclude_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let existing = std::fs::read_to_string(&exclude_path).unwrap_or_default();
    if existing
        .lines()
        .any(|line| line.trim() == ".lancedb/" || line.trim() == "/.lancedb/")
    {
        return Ok(());
    }

    let mut updated = existing;
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str("# zseek local index artifacts\n");
    updated.push_str(".lancedb/\n");
    std::fs::write(&exclude_path, updated)?;

    info!(
        "Added .lancedb/ to local git excludes: {}",
        exclude_path.display()
    );

    Ok(())
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
        ".expo",
        ".expo-shared",
        ".turbo",
        ".vercel",
        ".cache",
        ".parcel-cache",
        ".pnpm-store",
        ".yarn",
        ".npm",
        "coverage",
        ".nyc_output",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
    ];

    let ignored_file_names = [
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "composer.lock",
        "poetry.lock",
        "pdm.lock",
        "Pipfile.lock",
        "npm-debug.log",
        "yarn-error.log",
        "pnpm-debug.log",
    ];

    let ignored_path_fragments = [
        "/storage/logs/",
        "/storage/framework/",
        "/bootstrap/cache/",
        "/public/build/",
    ];

    for comp in path.components() {
        if let Some(comp_str) = comp.as_os_str().to_str() {
            if ignored_components.contains(&comp_str) {
                return true;
            }
        }
    }

    if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
        if ignored_file_names.contains(&file_name) {
            return true;
        }
    }

    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        let ext = ext.to_ascii_lowercase();
        if matches!(ext.as_str(), "log" | "tmp" | "pyc" | "pyo" | "pyd") {
            return true;
        }
    }

    let normalized = path.to_string_lossy().replace('\\', "/");
    let normalized = format!("/{}/", normalized.trim_matches('/'));
    for fragment in ignored_path_fragments {
        if normalized.contains(fragment) {
            return true;
        }
    }

    false
}

fn load_custom_ignore_matchers(workspace_roots: &[PathBuf]) -> Vec<WorkspaceIgnoreMatcher> {
    let mut matchers = Vec::new();

    for root in workspace_roots {
        let ignore_file = root.join(".zseekignore");
        if !ignore_file.is_file() {
            continue;
        }

        let mut builder = GitignoreBuilder::new(root);
        if let Some(err) = builder.add(&ignore_file) {
            warn!(
                "Failed reading custom ignore file {}: {}",
                ignore_file.display(),
                err
            );
            continue;
        }

        match builder.build() {
            Ok(matcher) => {
                info!("Loaded custom ignore file: {}", ignore_file.display());
                matchers.push(WorkspaceIgnoreMatcher {
                    root: root.clone(),
                    matcher,
                });
            }
            Err(err) => {
                warn!(
                    "Failed parsing custom ignore file {}: {}",
                    ignore_file.display(),
                    err
                );
            }
        }
    }

    matchers
}

fn is_ignored_path_with_matchers(path: &Path, matchers: &[WorkspaceIgnoreMatcher]) -> bool {
    if is_ignored_path(path) {
        return true;
    }

    let canonical = path.canonicalize().ok();
    let is_dir = path.is_dir();

    for matcher in matchers {
        let mut relative = None;

        if let Ok(value) = path.strip_prefix(&matcher.root) {
            relative = Some(value);
        }

        if relative.is_none() {
            if let Some(canonical_path) = canonical.as_ref() {
                if let Ok(value) = canonical_path.strip_prefix(&matcher.root) {
                    relative = Some(value);
                }
            }
        }

        if let Some(relative_path) = relative {
            let matched = matcher.matcher.matched(relative_path, is_dir);
            if matched.is_ignore() {
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
    ignore_matchers: Arc<Vec<WorkspaceIgnoreMatcher>>,
) {
    tokio::spawn(async move {
        let mut parser = CodeParser::new();
        let mut indexed_file_signatures = HashMap::<String, u64>::new();

        while let Some(event) = rx.recv().await {
            for path in event.paths {
                if is_ignored_path_with_matchers(&path, ignore_matchers.as_ref()) {
                    continue;
                }

                let path_key = path.to_string_lossy().to_string();

                if !path.exists() {
                    info!("File removed, purging stale chunks: {}", path.display());
                    if let Err(e) = db.replace_file_chunks(&path_key, Vec::new()).await {
                        error!("Failed to purge chunks for removed file {}: {}", path.display(), e);
                    }
                    indexed_file_signatures.remove(&path_key);
                    continue;
                }

                if !path.is_file() {
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

                let content_signature = file_content_signature(&content);
                if indexed_file_signatures
                    .get(&path_key)
                    .copied()
                    == Some(content_signature)
                {
                    continue;
                }

                let chunks = match parser.parse_file(&path, &content) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::debug!("Failed to parse {}: {}", path.display(), e);
                        continue;
                    }
                };

                if chunks.is_empty() {
                    if let Err(e) = db.replace_file_chunks(&path_key, Vec::new()).await {
                        error!(
                            "Failed to clear chunks for empty parse {}: {}",
                            path.display(),
                            e
                        );
                        continue;
                    }
                    indexed_file_signatures.insert(path_key, content_signature);
                    continue;
                }

                if !chunks.is_empty() {
                    info!(
                        "File changed: {} ({} chunks extracted)",
                        path.display(),
                        chunks.len()
                    );
                }

                let mut unique_chunks = Vec::new();
                let mut seen_chunks = HashSet::new();

                for chunk in chunks {
                    let signature = chunk_signature(&chunk);
                    if !seen_chunks.insert(signature) {
                        continue;
                    }

                    unique_chunks.push(chunk);
                }

                let embedding_inputs = unique_chunks
                    .iter()
                    .map(|chunk| chunk.content.clone())
                    .collect::<Vec<_>>();

                let embeddings = match provider.get_embeddings_batch(&embedding_inputs).await {
                    Ok(embeddings) => embeddings,
                    Err(e) => {
                        error!(
                            "Embedding batch failed for changed file {}: {}",
                            path.display(),
                            e
                        );
                        continue;
                    }
                };

                if embeddings.len() != unique_chunks.len() {
                    warn!(
                        "Skipping replacement for {} due to embedding count mismatch (chunks={}, embeddings={})",
                        path.display(),
                        unique_chunks.len(),
                        embeddings.len()
                    );
                    continue;
                }

                let chunks_buffer = unique_chunks
                    .into_iter()
                    .zip(embeddings.into_iter())
                    .collect::<Vec<_>>();

                info!(
                    "Replacing indexed chunks for {} with {} fresh chunks",
                    path.display(),
                    chunks_buffer.len()
                );

                if let Err(e) = db.replace_file_chunks(&path_key, chunks_buffer).await {
                    error!("Failed to replace file chunks for {}: {}", path.display(), e);
                    continue;
                }

                indexed_file_signatures.insert(path_key, content_signature);
            }
        }
    });
}

async fn initial_index(
    dir: &std::path::Path,
    provider: &Arc<CopilotProvider>,
    db: &Arc<VectorDb>,
    config: &WatchConfig,
    ignore_matchers: &[WorkspaceIgnoreMatcher],
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
        if !path.is_file() || is_ignored_path_with_matchers(path, ignore_matchers) {
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

        let path_key = path.to_string_lossy().to_string();

        if chunks.is_empty() {
            if let Err(e) = db.replace_file_chunks(&path_key, Vec::new()).await {
                error!("Failed clearing chunks for {}: {}", path.display(), e);
            }
            continue;
        }

        count += 1;
        info!(
            "Indexing file {}/{}: {}",
            count,
            config.max_file_count,
            path.display()
        );

        let mut file_buffer = Vec::new();
        let mut seen_chunk_signatures = HashSet::new();

        for chunk in chunks {
            let signature = chunk_signature(&chunk);
            if !seen_chunk_signatures.insert(signature) {
                continue;
            }

            file_buffer.push(chunk);
        }

        let embedding_inputs = file_buffer
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect::<Vec<_>>();

        let embeddings = match provider.get_embeddings_batch(&embedding_inputs).await {
            Ok(embeddings) => embeddings,
            Err(e) => {
                error!(
                    "Embedding batch failed during initial indexing for {}: {}",
                    path.display(),
                    e
                );
                continue;
            }
        };

        if embeddings.len() != file_buffer.len() {
            warn!(
                "Skipping replacement for {} due to embedding count mismatch (chunks={}, embeddings={})",
                path.display(),
                file_buffer.len(),
                embeddings.len()
            );
            continue;
        }

        let file_buffer = file_buffer
            .into_iter()
            .zip(embeddings.into_iter())
            .collect::<Vec<_>>();

        if let Err(e) = db.replace_file_chunks(&path_key, file_buffer).await {
            error!("Failed to replace file chunks for {}: {}", path.display(), e);
        }
    }

    info!(
        "Initial indexing complete for root {}. Indexed {} files.",
        dir.display(),
        count
    );
    Ok(())
}

fn file_content_signature(content: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

fn chunk_signature(chunk: &crate::parser::Chunk) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    chunk.file_path.hash(&mut hasher);
    chunk.start_line.hash(&mut hasher);
    chunk.end_line.hash(&mut hasher);
    chunk.content_hash.hash(&mut hasher);
    hasher.finish()
}

async fn run_benchmark(
    provider: &Arc<CopilotProvider>,
    db: &Arc<VectorDb>,
    limit: usize,
    cases_file: Option<&Path>,
    dataset_pack: Option<&str>,
    gate_config: &BenchmarkGateConfig,
    report_file: &Path,
) -> Result<()> {
    let cases = load_benchmark_cases(cases_file, dataset_pack)?;
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
    let mut precision_sum = 0.0f32;
    let mut mrr_sum = 0.0f32;
    let mut ndcg_sum = 0.0f32;

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

        precision_sum += precision_at_k_for_case(case, &results, top_k);
        mrr_sum += reciprocal_rank_for_case(case, &results, top_k);
        ndcg_sum += ndcg_at_k_for_case(case, &results, top_k);

        if let Some(distance) = results.first().and_then(|res| res.distance) {
            top_distance_sum += distance;
            top_distance_count += 1;
        }

        println!(
            "[{}] [{}] {} | hit={} | results={} | duplicates={}",
            idx + 1,
            case.task_family.as_deref().unwrap_or("uncategorized"),
            case.query,
            is_hit,
            results.len(),
            duplicates
        );
    }

    let total_cases = cases.len() as f32;
    let recall_at_k = hits as f32 / total_cases;
    let precision_at_k = precision_sum / total_cases;
    let mrr_at_k = mrr_sum / total_cases;
    let ndcg_at_k = ndcg_sum / total_cases;
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
    println!("Precision@{}: {:.3}", top_k, precision_at_k);
    println!("MRR@{}: {:.3}", top_k, mrr_at_k);
    println!("NDCG@{}: {:.3}", top_k, ndcg_at_k);
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
        precision_at_k,
        mrr_at_k,
        ndcg_at_k,
        no_result_rate,
        duplicate_rate,
        avg_top_distance,
    };

    let previous_report = load_previous_report(report_file)?;

    if let Some(previous) = previous_report.as_ref() {
        println!("\n=== Delta Vs Previous Run ===");
        print_metric_delta(
            &format!("Recall@{}", top_k),
            report.recall_at_k,
            previous.recall_at_k,
            false,
        );
        print_metric_delta(
            &format!("Precision@{}", top_k),
            report.precision_at_k,
            previous.precision_at_k,
            false,
        );
        print_metric_delta(
            &format!("MRR@{}", top_k),
            report.mrr_at_k,
            previous.mrr_at_k,
            false,
        );
        print_metric_delta(
            &format!("NDCG@{}", top_k),
            report.ndcg_at_k,
            previous.ndcg_at_k,
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

    if gate_config.enabled {
        let violations = evaluate_benchmark_gates(&report, previous_report.as_ref(), gate_config);
        println!("\n=== Quality Gates ===");
        if violations.is_empty() {
            println!("PASS");
        } else {
            for violation in &violations {
                println!("FAIL: {}", violation);
            }
            return Err(anyhow!(
                "Benchmark quality gates failed ({} violation(s))",
                violations.len()
            ));
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

fn evaluate_benchmark_gates(
    report: &BenchmarkReport,
    previous: Option<&BenchmarkReport>,
    config: &BenchmarkGateConfig,
) -> Vec<String> {
    if !config.enabled {
        return Vec::new();
    }

    let mut violations = Vec::new();

    if let Some(min_recall) = config.min_recall_at_k {
        if report.recall_at_k < min_recall {
            violations.push(format!(
                "Recall@{} {:.3} is below minimum {:.3}",
                report.top_k, report.recall_at_k, min_recall
            ));
        }
    }

    if let Some(min_precision) = config.min_precision_at_k {
        if report.precision_at_k < min_precision {
            violations.push(format!(
                "Precision@{} {:.3} is below minimum {:.3}",
                report.top_k, report.precision_at_k, min_precision
            ));
        }
    }

    if let Some(min_mrr) = config.min_mrr_at_k {
        if report.mrr_at_k < min_mrr {
            violations.push(format!(
                "MRR@{} {:.3} is below minimum {:.3}",
                report.top_k, report.mrr_at_k, min_mrr
            ));
        }
    }

    if let Some(min_ndcg) = config.min_ndcg_at_k {
        if report.ndcg_at_k < min_ndcg {
            violations.push(format!(
                "NDCG@{} {:.3} is below minimum {:.3}",
                report.top_k, report.ndcg_at_k, min_ndcg
            ));
        }
    }

    if let Some(max_no_result_rate) = config.max_no_result_rate {
        if report.no_result_rate > max_no_result_rate {
            violations.push(format!(
                "No-result rate {:.3} exceeds maximum {:.3}",
                report.no_result_rate, max_no_result_rate
            ));
        }
    }

    if let Some(max_duplicate_rate) = config.max_duplicate_rate {
        if report.duplicate_rate > max_duplicate_rate {
            violations.push(format!(
                "Duplicate rate {:.3} exceeds maximum {:.3}",
                report.duplicate_rate, max_duplicate_rate
            ));
        }
    }

    if config.require_non_regression {
        let Some(previous) = previous else {
            violations.push(
                "Non-regression gate is enabled but no previous benchmark report was found"
                    .to_string(),
            );
            return violations;
        };

        if previous.top_k != report.top_k {
            violations.push(format!(
                "Non-regression requires matching top-k (current={}, previous={})",
                report.top_k, previous.top_k
            ));
            return violations;
        }

        let tolerance = config.regression_tolerance.max(0.0);

        if report.recall_at_k + tolerance < previous.recall_at_k {
            violations.push(format!(
                "Recall@{} regressed from {:.3} to {:.3}",
                report.top_k, previous.recall_at_k, report.recall_at_k
            ));
        }

        if report.precision_at_k + tolerance < previous.precision_at_k {
            violations.push(format!(
                "Precision@{} regressed from {:.3} to {:.3}",
                report.top_k, previous.precision_at_k, report.precision_at_k
            ));
        }

        if report.mrr_at_k + tolerance < previous.mrr_at_k {
            violations.push(format!(
                "MRR@{} regressed from {:.3} to {:.3}",
                report.top_k, previous.mrr_at_k, report.mrr_at_k
            ));
        }

        if report.ndcg_at_k + tolerance < previous.ndcg_at_k {
            violations.push(format!(
                "NDCG@{} regressed from {:.3} to {:.3}",
                report.top_k, previous.ndcg_at_k, report.ndcg_at_k
            ));
        }

        if report.no_result_rate > previous.no_result_rate + tolerance {
            violations.push(format!(
                "No-result rate regressed from {:.3} to {:.3}",
                previous.no_result_rate, report.no_result_rate
            ));
        }

        if report.duplicate_rate > previous.duplicate_rate + tolerance {
            violations.push(format!(
                "Duplicate rate regressed from {:.3} to {:.3}",
                previous.duplicate_rate, report.duplicate_rate
            ));
        }
    }

    violations
}

fn benchmark_case(
    query: &str,
    expected_file_contains: &str,
    expected_symbol_contains: Option<&str>,
    task_family: &str,
) -> BenchmarkCase {
    BenchmarkCase {
        query: query.to_string(),
        expected_file_contains: expected_file_contains.to_string(),
        expected_symbol_contains: expected_symbol_contains.map(str::to_string),
        task_family: Some(task_family.to_string()),
    }
}

fn benchmark_pack_cases(pack_name: &str) -> Option<Vec<BenchmarkCase>> {
    let normalized = pack_name
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-")
        .replace(' ', "-");

    match normalized.as_str() {
        "core" => Some(vec![
            benchmark_case(
                "authentication token flow",
                "src/auth.rs",
                Some("run_auth_flow"),
                "core",
            ),
            benchmark_case(
                "vector database search query",
                "src/db.rs",
                Some("search"),
                "core",
            ),
            benchmark_case(
                "semantic search tool call handling",
                "src/mcp.rs",
                Some("rerank_chunks"),
                "core",
            ),
            benchmark_case(
                "code parser chunk extraction",
                "src/parser.rs",
                Some("parse_file"),
                "core",
            ),
            benchmark_case(
                "file watcher debounce behavior",
                "src/watcher.rs",
                Some("should_forward_event"),
                "core",
            ),
        ]),
        "bug-triage" => Some(vec![
            benchmark_case(
                "token refresh bug in auth flow",
                "src/auth.rs",
                Some("run_auth_flow"),
                "bug-triage",
            ),
            benchmark_case(
                "embedding retry failures in provider",
                "src/provider/github.rs",
                Some("get_embeddings_batch"),
                "bug-triage",
            ),
            benchmark_case(
                "stale chunk cleanup on file delete",
                "src/main.rs",
                Some("start_background_processor"),
                "bug-triage",
            ),
            benchmark_case(
                "workspace scope leakage filtering",
                "src/mcp.rs",
                Some("filter_results_to_scope"),
                "bug-triage",
            ),
        ]),
        "symbol-discovery" => Some(vec![
            benchmark_case(
                "find parse_file method",
                "src/parser.rs",
                Some("parse_file"),
                "symbol-discovery",
            ),
            benchmark_case(
                "find rerank profile function",
                "src/mcp.rs",
                Some("rerank_chunks_with_profile"),
                "symbol-discovery",
            ),
            benchmark_case(
                "find db search implementation",
                "src/db.rs",
                Some("search"),
                "symbol-discovery",
            ),
            benchmark_case(
                "find debounce helper",
                "src/watcher.rs",
                Some("should_forward_event"),
                "symbol-discovery",
            ),
        ]),
        "architecture-traversal" => Some(vec![
            benchmark_case(
                "mcp query flow to reranking",
                "src/mcp.rs",
                Some("run"),
                "architecture-traversal",
            ),
            benchmark_case(
                "watch pipeline parse embed replace",
                "src/main.rs",
                Some("start_background_processor"),
                "architecture-traversal",
            ),
            benchmark_case(
                "benchmark report and delta pipeline",
                "src/main.rs",
                Some("run_benchmark"),
                "architecture-traversal",
            ),
            benchmark_case(
                "workspace roots resolution flow",
                "src/workspace.rs",
                Some("resolve_scope_roots"),
                "architecture-traversal",
            ),
        ]),
        "cross-project-navigation" => Some(vec![
            benchmark_case(
                "provider embeddings to indexing handoff",
                "src/main.rs",
                Some("initial_index"),
                "cross-project-navigation",
            ),
            benchmark_case(
                "metadata extraction used for reranking",
                "src/parser.rs",
                Some("extract_symbol_metadata"),
                "cross-project-navigation",
            ),
            benchmark_case(
                "workspace file roots into scope filtering",
                "src/mcp.rs",
                Some("parse_scope_roots_from_initialize"),
                "cross-project-navigation",
            ),
            benchmark_case(
                "file replacement write path",
                "src/db.rs",
                Some("replace_file_chunks"),
                "cross-project-navigation",
            ),
        ]),
        "all" => {
            let mut all = Vec::new();
            for pack in [
                "bug-triage",
                "symbol-discovery",
                "architecture-traversal",
                "cross-project-navigation",
            ] {
                if let Some(mut cases) = benchmark_pack_cases(pack) {
                    all.append(&mut cases);
                }
            }
            Some(all)
        }
        _ => None,
    }
}

fn load_benchmark_cases(cases_file: Option<&Path>, dataset_pack: Option<&str>) -> Result<Vec<BenchmarkCase>> {
    if let Some(path) = cases_file {
        let raw = std::fs::read_to_string(path)?;
        let cases = serde_json::from_str::<Vec<BenchmarkCase>>(&raw)?;
        return Ok(cases);
    }

    let pack_name = dataset_pack.unwrap_or("core");
    if pack_name.eq_ignore_ascii_case("core") {
        return Ok(default_benchmark_cases());
    }

    benchmark_pack_cases(pack_name).ok_or_else(|| {
        anyhow!(
            "Unknown dataset pack '{}'. Available packs: core, bug-triage, symbol-discovery, architecture-traversal, cross-project-navigation, all",
            pack_name
        )
    })
}

fn default_benchmark_cases() -> Vec<BenchmarkCase> {
    benchmark_pack_cases("core").unwrap_or_default()
}

fn benchmark_hit(case: &BenchmarkCase, results: &[SearchResult]) -> bool {
    results
        .iter()
        .any(|result| relevance_score(case, result) > 0.0)
}

fn relevance_score(case: &BenchmarkCase, result: &SearchResult) -> f32 {
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

    if file_hit && symbol_hit {
        return 2.0;
    }

    if file_hit || symbol_hit {
        return 1.0;
    }

    0.0
}

fn reciprocal_rank_for_case(case: &BenchmarkCase, results: &[SearchResult], top_k: usize) -> f32 {
    let k = top_k.max(1).min(results.len());

    for (idx, result) in results.iter().take(k).enumerate() {
        if relevance_score(case, result) > 0.0 {
            return 1.0 / (idx as f32 + 1.0);
        }
    }

    0.0
}

fn precision_at_k_for_case(case: &BenchmarkCase, results: &[SearchResult], top_k: usize) -> f32 {
    let k = top_k.max(1);
    let relevant = results
        .iter()
        .take(k)
        .filter(|result| relevance_score(case, result) > 0.0)
        .count();

    relevant as f32 / k as f32
}

fn dcg_from_relevances(relevances: &[f32]) -> f32 {
    relevances
        .iter()
        .enumerate()
        .map(|(idx, rel)| {
            let gain = 2f32.powf(*rel) - 1.0;
            let discount = (idx as f32 + 2.0).log2();
            if discount <= 0.0 { 0.0 } else { gain / discount }
        })
        .sum()
}

fn ndcg_at_k_for_case(case: &BenchmarkCase, results: &[SearchResult], top_k: usize) -> f32 {
    let k = top_k.max(1).min(results.len());
    if k == 0 {
        return 0.0;
    }

    let mut relevances = results
        .iter()
        .take(k)
        .map(|result| relevance_score(case, result))
        .collect::<Vec<_>>();

    let dcg = dcg_from_relevances(&relevances);

    relevances.sort_by(|a, b| b.total_cmp(a));
    let idcg = dcg_from_relevances(&relevances);

    if idcg <= 0.0 {
        0.0
    } else {
        dcg / idcg
    }
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
    use super::{
        benchmark_hit, benchmark_pack_cases, chunk_signature, duplicate_count,
        evaluate_benchmark_gates, file_content_signature, load_benchmark_cases,
        format_workspace_roots_plan,
        ensure_local_git_exclude_for_workspace,
        is_ignored_path_with_matchers,
        is_ignored_path,
        load_custom_ignore_matchers,
        resolve_git_dir_for_workspace,
        ndcg_at_k_for_case, precision_at_k_for_case, reciprocal_rank_for_case, relevance_score,
        metric_trend, signed_delta, BenchmarkCase, BenchmarkGateConfig, BenchmarkReport, Cli,
        Commands,
    };
    use clap::Parser;
    use crate::db::SearchResult;
    use crate::parser::{content_hash_for_text, Chunk};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("{}-{}-{}", prefix, std::process::id(), suffix))
    }

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
                signature_fragment: None,
                visibility: None,
                arity: None,
                doc_comment_proximity: None,
                content_hash: content_hash_for_text("sample content"),
            },
            distance: Some(0.2),
        }
    }

    fn sample_report() -> BenchmarkReport {
        BenchmarkReport {
            generated_at_epoch_secs: 0,
            top_k: 5,
            total_cases: 10,
            hits: 7,
            no_result_cases: 1,
            total_results: 40,
            total_duplicates: 3,
            recall_at_k: 0.7,
            precision_at_k: 0.4,
            mrr_at_k: 0.62,
            ndcg_at_k: 0.67,
            no_result_rate: 0.1,
            duplicate_rate: 0.075,
            avg_top_distance: Some(0.32),
        }
    }

    #[test]
    fn benchmark_pack_cases_supports_named_task_families() {
        let bug_cases = benchmark_pack_cases("bug-triage").expect("pack should exist");
        assert!(!bug_cases.is_empty());
        assert!(bug_cases
            .iter()
            .all(|case| case.task_family.as_deref() == Some("bug-triage")));

        let all_cases = benchmark_pack_cases("all").expect("all pack should exist");
        assert!(all_cases.len() > bug_cases.len());
    }

    #[test]
    fn quality_gates_fail_when_absolute_thresholds_are_violated() {
        let report = sample_report();
        let config = BenchmarkGateConfig {
            enabled: true,
            require_non_regression: false,
            regression_tolerance: 0.0,
            min_recall_at_k: Some(0.8),
            min_precision_at_k: Some(0.5),
            min_mrr_at_k: None,
            min_ndcg_at_k: None,
            max_no_result_rate: Some(0.05),
            max_duplicate_rate: None,
        };

        let violations = evaluate_benchmark_gates(&report, None, &config);
        assert!(!violations.is_empty());
        assert!(violations
            .iter()
            .any(|item| item.contains("Recall@5") && item.contains("below minimum")));
        assert!(violations
            .iter()
            .any(|item| item.contains("No-result rate") && item.contains("exceeds maximum")));
    }

    #[test]
    fn quality_gates_detect_regressions_against_previous_report() {
        let current = sample_report();
        let mut previous = sample_report();
        previous.recall_at_k = 0.82;
        previous.precision_at_k = 0.55;
        previous.mrr_at_k = 0.72;
        previous.ndcg_at_k = 0.79;
        previous.no_result_rate = 0.04;
        previous.duplicate_rate = 0.03;

        let config = BenchmarkGateConfig {
            enabled: true,
            require_non_regression: true,
            regression_tolerance: 0.0,
            min_recall_at_k: None,
            min_precision_at_k: None,
            min_mrr_at_k: None,
            min_ndcg_at_k: None,
            max_no_result_rate: None,
            max_duplicate_rate: None,
        };

        let violations = evaluate_benchmark_gates(&current, Some(&previous), &config);
        assert!(violations
            .iter()
            .any(|item| item.contains("Recall@5 regressed")));
        assert!(violations
            .iter()
            .any(|item| item.contains("No-result rate regressed")));
    }

    #[test]
    fn quality_gates_require_matching_top_k_for_non_regression() {
        let current = sample_report();
        let mut previous = sample_report();
        previous.top_k = 10;

        let config = BenchmarkGateConfig {
            enabled: true,
            require_non_regression: true,
            regression_tolerance: 0.0,
            min_recall_at_k: None,
            min_precision_at_k: None,
            min_mrr_at_k: None,
            min_ndcg_at_k: None,
            max_no_result_rate: None,
            max_duplicate_rate: None,
        };

        let violations = evaluate_benchmark_gates(&current, Some(&previous), &config);
        assert!(violations
            .iter()
            .any(|item| item.contains("matching top-k")));
    }

    #[test]
    fn load_benchmark_cases_errors_for_unknown_dataset_pack() {
        let err = load_benchmark_cases(None, Some("nope-pack")).expect_err("pack should be rejected");
        let rendered = err.to_string();
        assert!(rendered.contains("Unknown dataset pack"));
        assert!(rendered.contains("nope-pack"));
    }

    #[test]
    fn benchmark_hit_matches_expected_file_or_symbol() {
        let case = BenchmarkCase {
            query: "auth flow".to_string(),
            expected_file_contains: "src/auth.rs".to_string(),
            expected_symbol_contains: Some("run_auth_flow".to_string()),
            task_family: None,
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

    #[test]
    fn file_content_signature_changes_with_content() {
        let a = file_content_signature("alpha");
        let b = file_content_signature("beta");
        assert_ne!(a, b);
    }

    #[test]
    fn format_workspace_roots_plan_enumerates_roots() {
        let roots = vec![
            PathBuf::from("/tmp/workspace/app"),
            PathBuf::from("/tmp/workspace/api"),
        ];

        let rendered = format_workspace_roots_plan(&roots);
        assert!(rendered.contains("1. /tmp/workspace/app"));
        assert!(rendered.contains("2. /tmp/workspace/api"));
    }

    #[test]
    fn format_workspace_roots_plan_handles_empty_input() {
        let rendered = format_workspace_roots_plan(&[]);
        assert_eq!(rendered, "(none)");
    }

    #[test]
    fn cli_parses_self_update_command() {
        let cli = Cli::parse_from(["zseek", "self-update"]);
        assert!(matches!(cli.command, Some(Commands::SelfUpdate)));
    }

    #[test]
    fn cli_parses_selfupdate_alias() {
        let cli = Cli::parse_from(["zseek", "selfupdate"]);
        assert!(matches!(cli.command, Some(Commands::SelfUpdate)));
    }

    #[test]
    fn ignore_rules_skip_log_files() {
        assert!(is_ignored_path(Path::new("storage/logs/laravel.log")));
    }

    #[test]
    fn ignore_rules_skip_javascript_framework_artifacts() {
        assert!(is_ignored_path(Path::new("apps/mobile/.expo/devices.json")));
        assert!(is_ignored_path(Path::new("apps/web/.next/cache/data.bin")));
        assert!(is_ignored_path(Path::new("apps/web/node_modules/react/index.js")));
    }

    #[test]
    fn ignore_rules_skip_laravel_runtime_artifacts() {
        assert!(is_ignored_path(Path::new("storage/framework/cache/data/meta")));
        assert!(is_ignored_path(Path::new("bootstrap/cache/services.php")));
    }

    #[test]
    fn ignore_rules_skip_python_runtime_artifacts() {
        assert!(is_ignored_path(Path::new(".venv/lib/python3.11/site.py")));
        assert!(is_ignored_path(Path::new(
            "src/__pycache__/app.cpython-311.pyc"
        )));
    }

    #[test]
    fn ignore_rules_keep_source_files_indexable() {
        assert!(!is_ignored_path(Path::new("src/main.rs")));
        assert!(!is_ignored_path(Path::new("app/Http/Controllers/UserController.php")));
    }

    #[test]
    fn custom_ignore_file_skips_configured_patterns() {
        let root = unique_temp_dir("zseek-custom-ignore");
        let src_dir = root.join("src");
        let logs_dir = root.join("logs");
        fs::create_dir_all(&src_dir).expect("create src dir");
        fs::create_dir_all(&logs_dir).expect("create logs dir");

        fs::write(root.join(".zseekignore"), "*.generated.ts\nlogs/**\n")
            .expect("write .zseekignore");

        let generated_file = src_dir.join("schema.generated.ts");
        let logs_file = logs_dir.join("worker.txt");
        let kept_file = src_dir.join("app.ts");

        fs::write(&generated_file, "export const schema = {};\n").expect("write generated file");
        fs::write(&logs_file, "runtime log\n").expect("write logs file");
        fs::write(&kept_file, "export const app = {};\n").expect("write source file");

        let matchers = load_custom_ignore_matchers(&[root.clone()]);
        assert!(is_ignored_path_with_matchers(&generated_file, &matchers));
        assert!(is_ignored_path_with_matchers(&logs_file, &matchers));
        assert!(!is_ignored_path_with_matchers(&kept_file, &matchers));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn custom_ignore_file_is_scoped_per_workspace_root() {
        let root_a = unique_temp_dir("zseek-ignore-scope-a");
        let root_b = unique_temp_dir("zseek-ignore-scope-b");
        let tmp_a = root_a.join("tmp");
        let tmp_b = root_b.join("tmp");
        fs::create_dir_all(&tmp_a).expect("create tmp in root a");
        fs::create_dir_all(&tmp_b).expect("create tmp in root b");

        fs::write(root_a.join(".zseekignore"), "tmp/**\n").expect("write root a ignore file");

        let ignored_in_a = tmp_a.join("cache.data");
        let kept_in_b = tmp_b.join("cache.data");
        fs::write(&ignored_in_a, "a\n").expect("write file in root a");
        fs::write(&kept_in_b, "b\n").expect("write file in root b");

        let matchers = load_custom_ignore_matchers(&[root_a.clone(), root_b.clone()]);
        assert!(is_ignored_path_with_matchers(&ignored_in_a, &matchers));
        assert!(!is_ignored_path_with_matchers(&kept_in_b, &matchers));

        let _ = fs::remove_dir_all(&root_a);
        let _ = fs::remove_dir_all(&root_b);
    }

    #[test]
    fn local_git_exclude_adds_lancedb_pattern_once() {
        let root = unique_temp_dir("zseek-git-exclude-add");
        let info_dir = root.join(".git").join("info");
        fs::create_dir_all(&info_dir).expect("create .git/info");

        let exclude_file = info_dir.join("exclude");
        fs::write(&exclude_file, "# existing rules\n").expect("write exclude file");

        ensure_local_git_exclude_for_workspace(&root).expect("first insert should succeed");
        ensure_local_git_exclude_for_workspace(&root).expect("second insert should be idempotent");

        let contents = fs::read_to_string(&exclude_file).expect("read exclude file");
        let lancedb_lines = contents
            .lines()
            .filter(|line| line.trim() == ".lancedb/")
            .count();

        assert_eq!(lancedb_lines, 1, "should not duplicate .lancedb/ entry");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_git_dir_supports_gitdir_pointer_file() {
        let root = unique_temp_dir("zseek-gitdir-pointer");
        let gitdir = root.join(".actual-git");
        fs::create_dir_all(gitdir.join("info")).expect("create gitdir info");
        fs::write(root.join(".git"), "gitdir: .actual-git\n").expect("write gitdir pointer");

        let resolved = resolve_git_dir_for_workspace(&root).expect("gitdir should resolve");
        assert_eq!(resolved, gitdir);

        ensure_local_git_exclude_for_workspace(&root).expect("exclude update should succeed");
        let exclude_contents =
            fs::read_to_string(root.join(".actual-git").join("info").join("exclude"))
                .expect("read exclude from gitdir");
        assert!(exclude_contents.lines().any(|line| line.trim() == ".lancedb/"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn chunk_signature_uses_path_span_and_content_hash_identity() {
        let mut first = sample_result("src/auth.rs", 10, 20, Some("run_auth_flow")).chunk;
        first.content = "old content".to_string();
        first.content_hash = "same-hash".to_string();

        let mut second = sample_result("src/auth.rs", 10, 20, Some("run_auth_flow")).chunk;
        second.content = "new content".to_string();
        second.content_hash = "same-hash".to_string();

        let first_signature = chunk_signature(&first);
        let second_signature = chunk_signature(&second);
        assert_eq!(first_signature, second_signature);

        second.content_hash = "different-hash".to_string();
        let third_signature = chunk_signature(&second);
        assert_ne!(first_signature, third_signature);
    }

    #[test]
    fn relevance_score_prioritizes_file_and_symbol_match() {
        let case = BenchmarkCase {
            query: "auth flow".to_string(),
            expected_file_contains: "src/auth.rs".to_string(),
            expected_symbol_contains: Some("run_auth_flow".to_string()),
            task_family: None,
        };

        let both = sample_result("src/auth.rs", 1, 5, Some("run_auth_flow"));
        let file_only = sample_result("src/auth.rs", 1, 5, Some("other_symbol"));
        let symbol_only = sample_result("src/other.rs", 1, 5, Some("run_auth_flow"));
        let none = sample_result("src/other.rs", 1, 5, Some("other_symbol"));

        assert!(relevance_score(&case, &both) > relevance_score(&case, &file_only));
        assert!(relevance_score(&case, &both) > relevance_score(&case, &symbol_only));
        assert_eq!(relevance_score(&case, &none), 0.0);
    }

    #[test]
    fn reciprocal_rank_uses_first_relevant_position() {
        let case = BenchmarkCase {
            query: "auth flow".to_string(),
            expected_file_contains: "src/auth.rs".to_string(),
            expected_symbol_contains: Some("run_auth_flow".to_string()),
            task_family: None,
        };

        let results = vec![
            sample_result("src/other.rs", 1, 5, Some("other")),
            sample_result("src/auth.rs", 10, 15, Some("run_auth_flow")),
            sample_result("src/auth.rs", 20, 30, Some("other")),
        ];

        let rr = reciprocal_rank_for_case(&case, &results, 5);
        assert!((rr - 0.5).abs() < 0.0001);
    }

    #[test]
    fn precision_at_k_uses_requested_cutoff() {
        let case = BenchmarkCase {
            query: "auth flow".to_string(),
            expected_file_contains: "src/auth.rs".to_string(),
            expected_symbol_contains: None,
            task_family: None,
        };

        let results = vec![
            sample_result("src/auth.rs", 1, 5, None),
            sample_result("src/other.rs", 6, 10, None),
            sample_result("src/auth.rs", 11, 15, None),
        ];

        let precision = precision_at_k_for_case(&case, &results, 5);
        assert!((precision - 0.4).abs() < 0.0001);
    }

    #[test]
    fn ndcg_rewards_earlier_relevance() {
        let case = BenchmarkCase {
            query: "auth flow".to_string(),
            expected_file_contains: "src/auth.rs".to_string(),
            expected_symbol_contains: Some("run_auth_flow".to_string()),
            task_family: None,
        };

        let better = vec![
            sample_result("src/auth.rs", 1, 5, Some("run_auth_flow")),
            sample_result("src/other.rs", 6, 10, Some("other")),
            sample_result("src/auth.rs", 11, 15, Some("other")),
        ];

        let worse = vec![
            sample_result("src/other.rs", 1, 5, Some("other")),
            sample_result("src/auth.rs", 6, 10, Some("other")),
            sample_result("src/auth.rs", 11, 15, Some("run_auth_flow")),
        ];

        let better_ndcg = ndcg_at_k_for_case(&case, &better, 5);
        let worse_ndcg = ndcg_at_k_for_case(&case, &worse, 5);
        assert!(better_ndcg > worse_ndcg);
    }
}
