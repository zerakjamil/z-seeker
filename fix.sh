cat > src/parser.rs << "EOF"
use anyhow::{anyhow, Result};
use std::path::Path;
use tree_sitter::Parser as TsParser;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub file_path: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

pub struct CodeParser {
    ts: TsParser,
}

impl CodeParser {
    pub fn new() -> Self {
        let ts = TsParser::new();
        Self { ts }
    }

    pub fn parse_file(&mut self, path: &Path, content: &str) -> Result<Vec<Chunk>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        
