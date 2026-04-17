use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;
use tree_sitter::Parser as TsParser;

const MIN_CHUNK_CHARS: usize = 50;
const MAX_CHUNK_CHARS: usize = 8000;
const LINE_CHUNK_SIZE: usize = 50;
const LINE_CHUNK_OVERLAP: usize = 10;
const MIN_ADAPTIVE_CHUNK_SIZE: usize = 24;
const MAX_ADAPTIVE_CHUNK_SIZE: usize = 80;
const MIN_ADAPTIVE_OVERLAP: usize = 6;
const MAX_ADAPTIVE_OVERLAP: usize = 20;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub file_path: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub signature_fragment: Option<String>,
    pub visibility: Option<String>,
    pub arity: Option<i32>,
    pub doc_comment_proximity: Option<i32>,
    pub content_hash: String,
}

#[derive(Debug, Clone, Default)]
struct SymbolMetadata {
    symbol_name: Option<String>,
    symbol_kind: Option<String>,
    signature_fragment: Option<String>,
    visibility: Option<String>,
    arity: Option<i32>,
    doc_comment_proximity: Option<i32>,
}

#[derive(Debug, Clone, Copy)]
struct LineChunkPlan {
    chunk_size: usize,
    overlap: usize,
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
        let file_lines: Vec<&str> = content.lines().collect();

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
                let metadata = Self::extract_symbol_metadata(node, content, &file_lines);

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
                        Some(&metadata),
                        Some(Self::node_depth(node)),
                    );
                    continue;
                }

                chunks.push(Chunk {
                    file_path: file_path.clone(),
                    content: text.to_string(),
                    start_line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: language_label.to_string(),
                    symbol_name: metadata.symbol_name,
                    symbol_kind: metadata.symbol_kind,
                    signature_fragment: metadata.signature_fragment,
                    visibility: metadata.visibility,
                    arity: metadata.arity,
                    doc_comment_proximity: metadata.doc_comment_proximity,
                    content_hash: content_hash_for_text(text),
                });
            }
        } else {
            Self::append_line_chunks(
                &mut chunks,
                &file_path,
                content,
                1,
                language_label,
                None,
                None,
            );
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
        lines: &[&str],
    ) -> SymbolMetadata {
        let symbol_kind = Self::map_symbol_kind(node);
        let symbol_name = Self::extract_identifier_from_node(node, content)
            .or_else(|| Self::extract_identifier_from_first_line(node, content));

        let declaration_text = content
            .get(node.start_byte()..node.end_byte())
            .unwrap_or_default();

        let signature_fragment = Self::signature_fragment(declaration_text);
        let visibility = signature_fragment
            .as_deref()
            .and_then(Self::extract_visibility);
        let arity = signature_fragment
            .as_deref()
            .and_then(Self::extract_arity);
        let start_line = node.start_position().row + 1;
        let doc_comment_proximity = Self::doc_comment_proximity(lines, start_line);

        SymbolMetadata {
            symbol_name,
            symbol_kind,
            signature_fragment,
            visibility,
            arity,
            doc_comment_proximity,
        }
    }

    fn signature_fragment(text: &str) -> Option<String> {
        let line = text
            .lines()
            .find(|line| !line.trim().is_empty())?
            .trim();
        if line.is_empty() {
            return None;
        }

        let compact = line
            .trim_end_matches('{')
            .trim_end_matches(';')
            .trim();

        if compact.is_empty() {
            None
        } else {
            Some(compact.chars().take(220).collect())
        }
    }

    fn extract_visibility(signature: &str) -> Option<String> {
        let normalized = signature.to_ascii_lowercase();
        if normalized.contains("pub(crate)") || normalized.contains("pub ( crate )") {
            Some("crate".to_string())
        } else if normalized.starts_with("pub ")
            || normalized.starts_with("export ")
            || normalized.contains(" public ")
        {
            Some("public".to_string())
        } else if normalized.contains(" protected ") {
            Some("protected".to_string())
        } else if normalized.contains(" private ") {
            Some("private".to_string())
        } else {
            Some("internal".to_string())
        }
    }

    fn extract_arity(signature: &str) -> Option<i32> {
        let open = signature.find('(')?;
        let close = signature[open + 1..].find(')')? + open + 1;
        let params = &signature[open + 1..close];
        if params.trim().is_empty() {
            return Some(0);
        }

        let count = params
            .split(',')
            .map(str::trim)
            .filter(|param| !param.is_empty())
            .filter(|param| {
                !matches!(
                    *param,
                    "self" | "&self" | "&mut self" | "this" | "cls" | "*args" | "**kwargs"
                )
            })
            .count();
        Some(count as i32)
    }

    fn doc_comment_proximity(lines: &[&str], declaration_start_line: usize) -> Option<i32> {
        if declaration_start_line <= 1 || lines.is_empty() {
            return None;
        }

        let mut cursor = declaration_start_line.saturating_sub(1) as i32;
        let mut scanned = 0usize;

        while cursor > 0 && scanned < 8 {
            let line = lines
                .get((cursor - 1) as usize)
                .map(|line| line.trim())
                .unwrap_or("");

            if line.is_empty() {
                cursor -= 1;
                scanned += 1;
                continue;
            }

            let is_doc_line = line.starts_with("///")
                || line.starts_with("//!")
                || line.starts_with("/**")
                || line.starts_with('*')
                || line.starts_with("##")
                || line.starts_with("\"\"\"")
                || line.starts_with("'''");

            if is_doc_line {
                let proximity = (declaration_start_line as i32 - cursor).max(0);
                return Some(proximity);
            }

            break;
        }

        None
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

    fn node_depth(node: tree_sitter::Node<'_>) -> usize {
        let mut depth = 0usize;
        let mut current = node.parent();

        while let Some(parent) = current {
            depth += 1;
            current = parent.parent();
        }

        depth
    }

    fn adaptive_line_chunk_plan(
        lines: &[&str],
        metadata: Option<&SymbolMetadata>,
        node_depth: Option<usize>,
    ) -> LineChunkPlan {
        if lines.is_empty() {
            return LineChunkPlan {
                chunk_size: LINE_CHUNK_SIZE,
                overlap: LINE_CHUNK_OVERLAP,
            };
        }

        let mut blank_lines = 0usize;
        let mut non_empty_lines = 0usize;
        let mut non_empty_chars = 0usize;
        let mut complexity_tokens = 0usize;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                blank_lines += 1;
                continue;
            }

            non_empty_lines += 1;
            non_empty_chars += trimmed.len();

            complexity_tokens += trimmed.matches('{').count();
            complexity_tokens += trimmed.matches('}').count();
            complexity_tokens += trimmed.matches('(').count();
            complexity_tokens += trimmed.matches(')').count();
            complexity_tokens += trimmed.matches("=>").count();
            complexity_tokens += trimmed.matches("::").count();

            for keyword in ["if ", "for ", "while ", "match ", "loop ", "else"] {
                if trimmed.contains(keyword) {
                    complexity_tokens += 1;
                }
            }
        }

        let line_count = lines.len();
        let blank_ratio = blank_lines as f32 / line_count.max(1) as f32;
        let avg_non_empty_len = if non_empty_lines > 0 {
            non_empty_chars as f32 / non_empty_lines as f32
        } else {
            0.0
        };
        let complexity_density = if non_empty_lines > 0 {
            complexity_tokens as f32 / non_empty_lines as f32
        } else {
            0.0
        };

        let mut target = LINE_CHUNK_SIZE as i32;

        if blank_ratio >= 0.35 && complexity_density <= 1.4 && avg_non_empty_len <= 40.0 {
            target += 20;
        }

        if complexity_density >= 3.0 {
            target -= 14;
        }

        if avg_non_empty_len >= 90.0 {
            target -= 8;
        }

        if let Some(metadata) = metadata {
            if matches!(
                metadata.symbol_kind.as_deref(),
                Some("function") | Some("method") | Some("closure")
            ) {
                target -= 4;
            }

            if metadata.arity.unwrap_or_default() >= 4 {
                target -= 4;
            }
        }

        if let Some(depth) = node_depth {
            if depth >= 5 {
                target -= 8;
            } else if depth >= 3 {
                target -= 4;
            }
        }

        if line_count <= LINE_CHUNK_SIZE {
            target = target.max(line_count as i32);
        }

        let chunk_size = target.clamp(
            MIN_ADAPTIVE_CHUNK_SIZE as i32,
            MAX_ADAPTIVE_CHUNK_SIZE as i32,
        ) as usize;

        let mut overlap = ((chunk_size as f32) * 0.2).round() as usize;
        if blank_ratio >= 0.35 {
            overlap += 2;
        }

        overlap = overlap.clamp(MIN_ADAPTIVE_OVERLAP, MAX_ADAPTIVE_OVERLAP);
        overlap = overlap.min(chunk_size.saturating_sub(1));

        LineChunkPlan {
            chunk_size,
            overlap,
        }
    }

    fn append_line_chunks(
        chunks: &mut Vec<Chunk>,
        file_path: &str,
        text: &str,
        base_start_line: usize,
        language: &str,
        metadata: Option<&SymbolMetadata>,
        node_depth: Option<usize>,
    ) {
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return;
        }

        let plan = Self::adaptive_line_chunk_plan(&lines, metadata, node_depth);
        let step = plan.chunk_size.saturating_sub(plan.overlap).max(1);
        let mut start_idx = 0usize;

        while start_idx < lines.len() {
            let end_idx = (start_idx + plan.chunk_size).min(lines.len());
            let chunk_lines = &lines[start_idx..end_idx];
            let chunk_text = chunk_lines.join("\n");

            if chunk_text.trim().len() >= MIN_CHUNK_CHARS {
                chunks.push(Chunk {
                    file_path: file_path.to_string(),
                    content: chunk_text,
                    start_line: base_start_line + start_idx,
                    end_line: base_start_line + end_idx - 1,
                    language: language.to_string(),
                    symbol_name: metadata.and_then(|m| m.symbol_name.clone()),
                    symbol_kind: metadata.and_then(|m| m.symbol_kind.clone()),
                    signature_fragment: metadata.and_then(|m| m.signature_fragment.clone()),
                    visibility: metadata.and_then(|m| m.visibility.clone()),
                    arity: metadata.and_then(|m| m.arity),
                    doc_comment_proximity: metadata.and_then(|m| m.doc_comment_proximity),
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
        assert!(
            first
                .signature_fragment
                .as_deref()
                .unwrap_or_default()
                .contains("fn first_method")
        );
        assert_eq!(first.visibility.as_deref(), Some("internal"));
        assert_eq!(first.arity, Some(0));
        assert!(!first.content_hash.is_empty());

        assert_eq!(second.language, "rust");
        assert_eq!(second.symbol_name.as_deref(), Some("second_method"));
        assert_eq!(second.symbol_kind.as_deref(), Some("method"));
        assert_eq!(second.arity, Some(0));
        assert!(!second.content_hash.is_empty());
    }

    #[test]
    fn parser_extracts_signature_visibility_and_doc_proximity() {
        let mut parser = CodeParser::new();
        let content = r#"
/// Handles login refresh
pub fn refresh_token(user: String, token: String) -> bool {
    !user.is_empty() && !token.is_empty()
}
"#;

        let chunks = parser
            .parse_file(Path::new("auth.rs"), content)
            .expect("rust code should parse");

        let chunk = chunks
            .iter()
            .find(|chunk| chunk.symbol_name.as_deref() == Some("refresh_token"))
            .expect("refresh_token chunk should exist");

        assert_eq!(chunk.symbol_kind.as_deref(), Some("function"));
        assert!(
            chunk
                .signature_fragment
                .as_deref()
                .unwrap_or_default()
                .contains("pub fn refresh_token")
        );
        assert_eq!(chunk.visibility.as_deref(), Some("public"));
        assert_eq!(chunk.arity, Some(2));
        assert_eq!(chunk.doc_comment_proximity, Some(1));
    }

    #[test]
    fn adaptive_chunking_uses_smaller_windows_for_deep_complex_nodes() {
        let mut parser = CodeParser::new();
        let mut lines = vec![
            "pub fn monster_flow(input: usize) -> usize {".to_string(),
            "    let mut acc = input;".to_string(),
        ];

        for idx in 0..260 {
            lines.push(format!(
                "    if acc % 2 == 0 {{ acc = acc.wrapping_add({}); }} else {{ acc = acc.wrapping_mul({}); }} // branch {} {}",
                idx + 3,
                idx + 5,
                idx,
                "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            ));
        }

        lines.push("    acc".to_string());
        lines.push("}".to_string());

        let content = lines.join("\n");
        let chunks = parser
            .parse_file(Path::new("complex.rs"), &content)
            .expect("complex rust function should parse");

        let monster_chunks = chunks
            .iter()
            .filter(|chunk| chunk.symbol_name.as_deref() == Some("monster_flow"))
            .collect::<Vec<_>>();

        assert!(
            monster_chunks.len() >= 3,
            "expected oversized semantic node to be split into multiple chunks"
        );
        assert!(
            monster_chunks
                .iter()
                .all(|chunk| (chunk.end_line - chunk.start_line + 1) <= 40),
            "adaptive chunking should reduce chunk windows for deep and complex nodes"
        );
    }

    #[test]
    fn adaptive_chunking_expands_windows_for_sparse_text() {
        let mut parser = CodeParser::new();
        let content = (1..=120)
            .map(|line| {
                if line % 2 == 0 {
                    "".to_string()
                } else {
                    format!("note line {:03}", line)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let chunks = parser
            .parse_file(Path::new("notes.txt"), &content)
            .expect("text fallback should parse");

        assert_eq!(
            chunks.len(),
            2,
            "sparse text should use larger chunk windows to improve recall"
        );
        assert!(
            chunks[0].end_line > 50,
            "first sparse chunk should exceed the fixed 50-line baseline"
        );
    }
}
