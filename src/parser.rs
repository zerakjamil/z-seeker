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
        
        let language = match ext {
            "rs" => tree_sitter_rust::LANGUAGE.into(),
            "ts" | "tsx" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            _ => return Err(anyhow!("Unsupported language extension: {}", ext)),
        };

        self.ts.set_language(&language)?;
        
        let tree = self.ts.parse(content, None).ok_or_else(|| anyhow!("Failed to parse file AST"))?;
        
        let mut chunks = Vec::new();
        let mut cursor = tree.walk();
        let root = tree.root_node();

        for child in root.children(&mut cursor) {
            let start_byte = child.start_byte();
            let end_byte = child.end_byte();

            if end_byte <= start_byte {
                continue;
            }

            let text = &content[start_byte..end_byte];
            
            if text.len() < 50 {
                continue;
            }

            chunks.push(Chunk {
                file_path: path.to_string_lossy().to_string(),
                content: text.to_string(),
                start_line: child.start_position().row + 1,
                end_line: child.end_position().row + 1,
            });
        }

        Ok(chunks)
    }
}