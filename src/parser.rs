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
            "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
            "ts" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
            "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
            "py" => Some(tree_sitter_python::LANGUAGE.into()),
            "java" => Some(tree_sitter_java::LANGUAGE.into()),
            "go" => Some(tree_sitter_go::LANGUAGE.into()),
            "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
            "cpp" | "hpp" | "cc" | "cxx" => Some(tree_sitter_cpp::LANGUAGE.into()),
            "json" => Some(tree_sitter_json::LANGUAGE.into()),
            "sh" | "bash" => Some(tree_sitter_bash::LANGUAGE.into()),
            "md" | "txt" | "yaml" | "yml" | "toml" | "html" | "css" => None,
            _ => return Err(anyhow!("Unsupported language extension: {}", ext)),
        };

        let mut chunks = Vec::new();

        if let Some(language) = language {
            self.ts.set_language(&language)?;

            let tree = self
                .ts
                .parse(content, None)
                .ok_or_else(|| anyhow!("Failed to parse file AST"))?;
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
        } else {
            let lines: Vec<&str> = content.lines().collect();
            let chunk_size = 50;
            for (i, chunk_lines) in lines.chunks(chunk_size).enumerate() {
                let text = chunk_lines.join("\n");
                if text.trim().len() < 50 {
                    continue;
                }
                chunks.push(Chunk {
                    file_path: path.to_string_lossy().to_string(),
                    content: text,
                    start_line: i * chunk_size + 1,
                    end_line: i * chunk_size + chunk_lines.len(),
                });
            }
        }

        Ok(chunks)
    }
}
