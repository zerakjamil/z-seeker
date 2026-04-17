use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;
use tree_sitter::Parser as TsParser;

const MIN_CHUNK_CHARS: usize = 50;
const MAX_CHUNK_CHARS: usize = 8000;
const LINE_CHUNK_SIZE: usize = 50;
const LINE_CHUNK_OVERLAP: usize = 10;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub file_path: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub content_hash: String,
}

pub fn content_hash_for_text(content: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
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
        let file_path = path.to_string_lossy().to_string();

        let (language, language_label) = match ext {
            "rs" => (Some(tree_sitter_rust::LANGUAGE.into()), "rust"),
            "ts" | "tsx" => (Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()), "typescript"),
            "js" | "jsx" => (Some(tree_sitter_javascript::LANGUAGE.into()), "javascript"),
            "php" => (Some(tree_sitter_php::LANGUAGE_PHP.into()), "php"),
            "py" => (Some(tree_sitter_python::LANGUAGE.into()), "python"),
            "java" => (Some(tree_sitter_java::LANGUAGE.into()), "java"),
            "go" => (Some(tree_sitter_go::LANGUAGE.into()), "go"),
            "c" | "h" => (Some(tree_sitter_c::LANGUAGE.into()), "c"),
            "cpp" | "hpp" | "cc" | "cxx" => (Some(tree_sitter_cpp::LANGUAGE.into()), "cpp"),
            "json" => (Some(tree_sitter_json::LANGUAGE.into()), "json"),
            "sh" | "bash" => (Some(tree_sitter_bash::LANGUAGE.into()), "bash"),
            "md" | "txt" | "yaml" | "yml" | "toml" | "html" | "css" => (None, "text"),
            _ => (None, "text"),
        };

        let mut chunks = Vec::new();

        if let Some(language) = language {
            self.ts.set_language(&language)?;

            let tree = self
                .ts
                .parse(content, None)
                .ok_or_else(|| anyhow!("Failed to parse file AST"))?;
            let root = tree.root_node();
            let mut semantic_nodes = Self::collect_semantic_nodes(root);

            if semantic_nodes.is_empty() {
                let mut cursor = root.walk();
                for child in root.named_children(&mut cursor) {
                    semantic_nodes.push(child);
                }
            }

            if semantic_nodes.is_empty() {
                semantic_nodes.push(root);
            }

            semantic_nodes.sort_by_key(|n| n.start_byte());
            let mut seen_spans = HashSet::new();

            for node in semantic_nodes {
                let start_byte = node.start_byte();
                let end_byte = node.end_byte();

                if end_byte <= start_byte {
                    continue;
                }

                if !seen_spans.insert((start_byte, end_byte)) {
                    continue;
                }

                let text = &content[start_byte..end_byte];
                let (symbol_name, symbol_kind) = Self::extract_symbol_metadata(node, content);

                if text.trim().len() < MIN_CHUNK_CHARS {
                    continue;
                }

                if text.len() > MAX_CHUNK_CHARS {
                    // Fallback to line chunking if a single AST node is too massive
                    Self::append_line_chunks(
                        &mut chunks,
                        &file_path,
                        text,
                        node.start_position().row + 1,
                        language_label,
                        symbol_name.clone(),
                        symbol_kind.clone(),
                    );
                    continue;
                }

                chunks.push(Chunk {
                    file_path: file_path.clone(),
                    content: text.to_string(),
                    start_line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: language_label.to_string(),
                    symbol_name,
                    symbol_kind,
                    content_hash: content_hash_for_text(text),
                });
            }
        } else {
            Self::append_line_chunks(&mut chunks, &file_path, content, 1, language_label, None, None);
        }

        Ok(chunks)
    }

    fn collect_semantic_nodes<'tree>(root: tree_sitter::Node<'tree>) -> Vec<tree_sitter::Node<'tree>> {
        let mut stack = vec![root];
        let mut nodes = Vec::new();

        while let Some(node) = stack.pop() {
            if Self::is_semantic_node(node.kind()) {
                nodes.push(node);
                continue;
            }

            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                stack.push(child);
            }
        }

        nodes
    }

    fn is_semantic_node(kind: &str) -> bool {
        matches!(
            kind,
            "function_item"
                | "function_definition"
                | "function_declaration"
                | "method_definition"
                | "method_declaration"
                | "constructor_declaration"
                | "lambda_expression"
                | "closure_expression"
                | "class_declaration"
                | "class_definition"
                | "struct_item"
                | "interface_declaration"
                | "enum_declaration"
        )
    }

    fn extract_symbol_metadata(
        node: tree_sitter::Node<'_>,
        content: &str,
    ) -> (Option<String>, Option<String>) {
        let symbol_kind = Self::map_symbol_kind(node);
        let symbol_name = Self::extract_identifier_from_node(node, content)
            .or_else(|| Self::extract_identifier_from_first_line(node, content));

        (symbol_name, symbol_kind)
    }

    fn map_symbol_kind(node: tree_sitter::Node<'_>) -> Option<String> {
        let node_kind = node.kind();
        let mapped = match node_kind {
            "function_item" => {
                if Self::is_within_parent(node, &["impl_item", "impl_block", "impl_body"]) {
                    "method"
                } else {
                    "function"
                }
            }
            "function_definition" | "function_declaration" => "function",
            "method_definition" | "method_declaration" | "constructor_declaration" => "method",
            "class_declaration" | "class_definition" => "class",
            "struct_item" => "struct",
            "enum_declaration" => "enum",
            "interface_declaration" => "interface",
            "lambda_expression" | "closure_expression" => "closure",
            _ => return None,
        };

        Some(mapped.to_string())
    }

    fn is_within_parent(node: tree_sitter::Node<'_>, parent_kinds: &[&str]) -> bool {
        let mut current = node.parent();
        while let Some(parent) = current {
            if parent_kinds.contains(&parent.kind()) {
                return true;
            }
            current = parent.parent();
        }
        false
    }

    fn extract_identifier_from_node(node: tree_sitter::Node<'_>, content: &str) -> Option<String> {
        let identifier_kinds = [
            "identifier",
            "field_identifier",
            "property_identifier",
            "type_identifier",
            "name",
            "word",
        ];

        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            let mut cursor = current.walk();
            for child in current.named_children(&mut cursor) {
                if identifier_kinds.contains(&child.kind()) {
                    if let Some(name) = content
                        .get(child.start_byte()..child.end_byte())
                        .map(str::trim)
                        .filter(|name| !name.is_empty())
                    {
                        return Some(name.to_string());
                    }
                }

                stack.push(child);
            }
        }

        None
    }

    fn extract_identifier_from_first_line(
        node: tree_sitter::Node<'_>,
        content: &str,
    ) -> Option<String> {
        let text = content.get(node.start_byte()..node.end_byte())?;
        let first_line = text.lines().next()?.trim();
        let prefixes = ["fn ", "function ", "class ", "struct ", "enum ", "trait ", "impl "];

        for prefix in prefixes {
            if let Some(rest) = first_line.strip_prefix(prefix) {
                let identifier: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !identifier.is_empty() {
                    return Some(identifier);
                }
            }
        }

        None
    }

    fn append_line_chunks(
        chunks: &mut Vec<Chunk>,
        file_path: &str,
        text: &str,
        base_start_line: usize,
        language: &str,
        symbol_name: Option<String>,
        symbol_kind: Option<String>,
    ) {
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return;
        }

        let step = LINE_CHUNK_SIZE.saturating_sub(LINE_CHUNK_OVERLAP).max(1);
        let mut start_idx = 0usize;

        while start_idx < lines.len() {
            let end_idx = (start_idx + LINE_CHUNK_SIZE).min(lines.len());
            let chunk_lines = &lines[start_idx..end_idx];
            let chunk_text = chunk_lines.join("\n");

            if chunk_text.trim().len() >= MIN_CHUNK_CHARS {
                chunks.push(Chunk {
                    file_path: file_path.to_string(),
                    content: chunk_text,
                    start_line: base_start_line + start_idx,
                    end_line: base_start_line + end_idx - 1,
                    language: language.to_string(),
                    symbol_name: symbol_name.clone(),
                    symbol_kind: symbol_kind.clone(),
                    content_hash: content_hash_for_text(&chunk_lines.join("\n")),
                });
            }

            if end_idx == lines.len() {
                break;
            }
            start_idx += step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CodeParser;
    use std::path::Path;

    fn make_line(index: usize) -> String {
        format!(
            "line {:03} - this is deliberately verbose content for stable chunk tests",
            index
        )
    }

    fn make_lines(count: usize) -> String {
        (1..=count)
            .map(make_line)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parse_unknown_extension_falls_back_to_text_chunking() {
        let mut parser = CodeParser::new();
        let content = make_lines(8);

        let chunks = parser
            .parse_file(Path::new("notes.customext"), &content)
            .expect("unknown extensions should use text fallback");

        assert!(!chunks.is_empty(), "expected at least one fallback chunk");
        assert!(chunks.iter().all(|c| c.language == "text"));
        assert!(chunks.iter().all(|c| c.symbol_name.is_none()));
        assert!(chunks.iter().all(|c| !c.content_hash.is_empty()));
    }

    #[test]
    fn text_chunking_uses_overlap_for_better_context_continuity() {
        let mut parser = CodeParser::new();
        let content = make_lines(70);

        let chunks = parser
            .parse_file(Path::new("notes.txt"), &content)
            .expect("text chunking should parse successfully");

        assert_eq!(chunks.len(), 2, "70 lines should produce two overlapping chunks");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 50);
        assert_eq!(
            chunks[1].start_line, 41,
            "second chunk should overlap by 10 lines"
        );
        assert_eq!(chunks[1].end_line, 70);
    }

    #[test]
    fn rust_impl_methods_are_chunked_individually_for_precision() {
        let mut parser = CodeParser::new();
        let content = r#"
impl Demo {
    fn first_method(&self) {
        let _x = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    }

    fn second_method(&self) {
        let _y = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    }
}
"#;

        let chunks = parser
            .parse_file(Path::new("demo.rs"), content)
            .expect("rust code should parse successfully");

        assert!(
            chunks.len() >= 2,
            "expected method-level chunks for nested impl methods"
        );
        let first = chunks
            .iter()
            .find(|c| c.content.contains("fn first_method"))
            .expect("first method chunk should exist");
        let second = chunks
            .iter()
            .find(|c| c.content.contains("fn second_method"))
            .expect("second method chunk should exist");

        assert_eq!(first.language, "rust");
        assert_eq!(first.symbol_name.as_deref(), Some("first_method"));
        assert_eq!(first.symbol_kind.as_deref(), Some("method"));
        assert!(!first.content_hash.is_empty());

        assert_eq!(second.language, "rust");
        assert_eq!(second.symbol_name.as_deref(), Some("second_method"));
        assert_eq!(second.symbol_kind.as_deref(), Some("method"));
        assert!(!second.content_hash.is_empty());
    }
}
