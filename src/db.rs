use crate::parser::{content_hash_for_text, Chunk};
use anyhow::{anyhow, Result};
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int32Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::StreamExt;
use lancedb::{connect, Table};
use std::sync::Arc;
use tracing::info;

use lancedb::query::{ExecutableQuery, QueryBase};
pub struct VectorDb {
    table: Table,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub distance: Option<f32>,
}

impl VectorDb {
    pub async fn new(uri: &str) -> Result<Self> {
        let conn = connect(uri).execute().await?;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("start_line", DataType::Int32, false),
            Field::new("end_line", DataType::Int32, false),
            Field::new("language", DataType::Utf8, false),
            Field::new("symbol_name", DataType::Utf8, true),
            Field::new("symbol_kind", DataType::Utf8, true),
            Field::new("signature_fragment", DataType::Utf8, true),
            Field::new("visibility", DataType::Utf8, true),
            Field::new("arity", DataType::Int32, true),
            Field::new("doc_comment_proximity", DataType::Int32, true),
            Field::new("content_hash", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    1536,
                ),
                false,
            ),
        ]));

        let table_name = "semantic_chunks";
        let table = match conn.open_table(table_name).execute().await {
            Ok(t) => t,
            Err(_) => {
                info!("Creating LanceDB table '{}' at {}", table_name, uri);
                conn.create_empty_table(table_name, schema)
                    .execute()
                    .await?
            }
        };

        Ok(Self { table })
    }

    pub async fn add_chunks(&self, data: Vec<(Chunk, Vec<f32>)>) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let schema = self.table.schema().await?;
        let has_core_metadata_columns = schema.index_of("language").is_ok()
            && schema.index_of("symbol_name").is_ok()
            && schema.index_of("symbol_kind").is_ok()
            && schema.index_of("content_hash").is_ok();
        let has_extended_metadata_columns = has_core_metadata_columns
            && schema.index_of("signature_fragment").is_ok()
            && schema.index_of("visibility").is_ok()
            && schema.index_of("arity").is_ok()
            && schema.index_of("doc_comment_proximity").is_ok();

        let mut file_paths = Vec::with_capacity(data.len());
        let mut contents = Vec::with_capacity(data.len());
        let mut start_lines = Vec::with_capacity(data.len());
        let mut end_lines = Vec::with_capacity(data.len());
        let mut languages = Vec::with_capacity(data.len());
        let mut symbol_names = Vec::with_capacity(data.len());
        let mut symbol_kinds = Vec::with_capacity(data.len());
        let mut signature_fragments = Vec::with_capacity(data.len());
        let mut visibilities = Vec::with_capacity(data.len());
        let mut arities = Vec::with_capacity(data.len());
        let mut doc_comment_proximities = Vec::with_capacity(data.len());
        let mut content_hashes = Vec::with_capacity(data.len());
        let mut vectors = Vec::with_capacity(data.len());

        for (chunk, embedding) in data {
            file_paths.push(chunk.file_path);
            contents.push(chunk.content);
            start_lines.push(chunk.start_line as i32);
            end_lines.push(chunk.end_line as i32);
            languages.push(chunk.language);
            symbol_names.push(chunk.symbol_name);
            symbol_kinds.push(chunk.symbol_kind);
            signature_fragments.push(chunk.signature_fragment);
            visibilities.push(chunk.visibility);
            arities.push(chunk.arity);
            doc_comment_proximities.push(chunk.doc_comment_proximity);
            content_hashes.push(chunk.content_hash);
            vectors.push(Some(embedding.into_iter().map(Some).collect::<Vec<_>>()));
        }

        let file_path_array = StringArray::from(file_paths);
        let content_array = StringArray::from(contents);
        let start_line_array = Int32Array::from(start_lines);
        let end_line_array = Int32Array::from(end_lines);
        let language_array = StringArray::from(languages);
        let symbol_name_array = StringArray::from(symbol_names);
        let symbol_kind_array = StringArray::from(symbol_kinds);
        let signature_fragment_array = StringArray::from(signature_fragments);
        let visibility_array = StringArray::from(visibilities);
        let arity_array = Int32Array::from(arities);
        let doc_comment_proximity_array = Int32Array::from(doc_comment_proximities);
        let content_hash_array = StringArray::from(content_hashes);
        let vector_array =
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, 1536);

        let batch = if has_extended_metadata_columns {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(language_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_name_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_kind_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(signature_fragment_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(visibility_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(arity_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(doc_comment_proximity_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_hash_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
                ],
            )?
        } else if has_core_metadata_columns {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(language_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_name_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_kind_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_hash_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
                ],
            )?
        } else {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
                ],
            )?
        };

        let batches = vec![Ok(batch)];
        let iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());
        self.table.add(iter).execute().await?;
        Ok(())
    }

    pub async fn replace_file_chunks(
        &self,
        file_path: &str,
        data: Vec<(Chunk, Vec<f32>)>,
    ) -> Result<()> {
        let escaped_file_path = file_path.replace('\'', "''");
        let predicate = format!("file_path = '{}'", escaped_file_path);
        self.table.delete(&predicate).await?;

        if data.is_empty() {
            return Ok(());
        }

        self.add_chunks(data).await
    }

    pub async fn add_chunk(&self, chunk: Chunk, embedding: Vec<f32>) -> Result<()> {
        let schema = self.table.schema().await?;
        let has_core_metadata_columns = schema.index_of("language").is_ok()
            && schema.index_of("symbol_name").is_ok()
            && schema.index_of("symbol_kind").is_ok()
            && schema.index_of("content_hash").is_ok();
        let has_extended_metadata_columns = has_core_metadata_columns
            && schema.index_of("signature_fragment").is_ok()
            && schema.index_of("visibility").is_ok()
            && schema.index_of("arity").is_ok()
            && schema.index_of("doc_comment_proximity").is_ok();

        let file_path_array = StringArray::from(vec![chunk.file_path.clone()]);
        let content_array = StringArray::from(vec![chunk.content.clone()]);
        let start_line_array = Int32Array::from(vec![chunk.start_line as i32]);
        let end_line_array = Int32Array::from(vec![chunk.end_line as i32]);
        let language_array = StringArray::from(vec![chunk.language.clone()]);
        let symbol_name_array = StringArray::from(vec![chunk.symbol_name.clone()]);
        let symbol_kind_array = StringArray::from(vec![chunk.symbol_kind.clone()]);
        let signature_fragment_array = StringArray::from(vec![chunk.signature_fragment.clone()]);
        let visibility_array = StringArray::from(vec![chunk.visibility.clone()]);
        let arity_array = Int32Array::from(vec![chunk.arity]);
        let doc_comment_proximity_array = Int32Array::from(vec![chunk.doc_comment_proximity]);
        let content_hash_array = StringArray::from(vec![chunk.content_hash.clone()]);

        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(embedding.into_iter().map(Some).collect::<Vec<_>>())],
            1536,
        );

        let batch = if has_extended_metadata_columns {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(language_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_name_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_kind_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(signature_fragment_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(visibility_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(arity_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(doc_comment_proximity_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_hash_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
                ],
            )?
        } else if has_core_metadata_columns {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(language_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_name_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(symbol_kind_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_hash_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
                ],
            )?
        } else {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                    Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
                ],
            )?
        };

        // LanceDB 0.14 IntoArrow is implemented for RecordBatchIterator directly
        let batches = vec![Ok(batch)];
        let iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());
        self.table.add(iter).execute().await?;
        Ok(())
    }

    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        limit_count: usize,
    ) -> Result<Vec<SearchResult>> {
        let qs = query_embedding.as_slice();
        let mut results = self
            .table
            .query()
            .nearest_to(qs)?
            .limit(limit_count)
            .execute()
            .await?;

        let mut chunks = Vec::new();

        while let Some(batch_res) = results.next().await {
            let batch = batch_res?;
            if batch.num_rows() == 0 {
                continue;
            }

            let schema = batch.schema();
            let file_path_idx = schema
                .index_of("file_path")
                .map_err(|_| anyhow!("Missing file_path column in search results"))?;
            let content_idx = schema
                .index_of("content")
                .map_err(|_| anyhow!("Missing content column in search results"))?;
            let start_line_idx = schema
                .index_of("start_line")
                .map_err(|_| anyhow!("Missing start_line column in search results"))?;
            let end_line_idx = schema
                .index_of("end_line")
                .map_err(|_| anyhow!("Missing end_line column in search results"))?;
            let language_idx = schema.index_of("language").ok();
            let symbol_name_idx = schema.index_of("symbol_name").ok();
            let symbol_kind_idx = schema.index_of("symbol_kind").ok();
            let signature_fragment_idx = schema.index_of("signature_fragment").ok();
            let visibility_idx = schema.index_of("visibility").ok();
            let arity_idx = schema.index_of("arity").ok();
            let doc_comment_proximity_idx = schema.index_of("doc_comment_proximity").ok();
            let content_hash_idx = schema.index_of("content_hash").ok();
            let distance_idx = schema.index_of("_distance").ok();

            let file_paths = batch
                .column(file_path_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("file_path column has unexpected type"))?;
            let contents = batch
                .column(content_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("content column has unexpected type"))?;
            let start_lines = batch
                .column(start_line_idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| anyhow!("start_line column has unexpected type"))?;
            let end_lines = batch
                .column(end_line_idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| anyhow!("end_line column has unexpected type"))?;

            let language_values = language_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
            });
            let symbol_name_values = symbol_name_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
            });
            let symbol_kind_values = symbol_kind_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
            });
            let signature_fragment_values = signature_fragment_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
            });
            let visibility_values = visibility_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
            });
            let arity_values = arity_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<Int32Array>()
            });
            let doc_comment_proximity_values = doc_comment_proximity_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<Int32Array>()
            });
            let content_hash_values = content_hash_idx.and_then(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
            });

            for i in 0..batch.num_rows() {
                let language = language_values
                    .map(|values| {
                        if values.is_null(i) {
                            "unknown".to_string()
                        } else {
                            values.value(i).to_string()
                        }
                    })
                    .unwrap_or_else(|| "unknown".to_string());

                let symbol_name = symbol_name_values.and_then(|values| {
                    if values.is_null(i) {
                        None
                    } else {
                        Some(values.value(i).to_string())
                    }
                });

                let symbol_kind = symbol_kind_values.and_then(|values| {
                    if values.is_null(i) {
                        None
                    } else {
                        Some(values.value(i).to_string())
                    }
                });

                let signature_fragment = signature_fragment_values.and_then(|values| {
                    if values.is_null(i) {
                        None
                    } else {
                        Some(values.value(i).to_string())
                    }
                });

                let visibility = visibility_values.and_then(|values| {
                    if values.is_null(i) {
                        None
                    } else {
                        Some(values.value(i).to_string())
                    }
                });

                let arity = arity_values.and_then(|values| {
                    if values.is_null(i) {
                        None
                    } else {
                        Some(values.value(i))
                    }
                });

                let doc_comment_proximity = doc_comment_proximity_values.and_then(|values| {
                    if values.is_null(i) {
                        None
                    } else {
                        Some(values.value(i))
                    }
                });

                let content = contents.value(i).to_string();
                let content_hash = content_hash_values
                    .map(|values| {
                        if values.is_null(i) {
                            content_hash_for_text(&content)
                        } else {
                            values.value(i).to_string()
                        }
                    })
                    .unwrap_or_else(|| content_hash_for_text(&content));

                let distance = distance_idx.and_then(|idx| {
                    let array = batch.column(idx);

                    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
                        if values.is_null(i) {
                            None
                        } else {
                            Some(values.value(i))
                        }
                    } else if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
                        if values.is_null(i) {
                            None
                        } else {
                            Some(values.value(i) as f32)
                        }
                    } else {
                        None
                    }
                });

                chunks.push(SearchResult {
                    chunk: Chunk {
                        file_path: file_paths.value(i).to_string(),
                        content,
                        start_line: start_lines.value(i) as usize,
                        end_line: end_lines.value(i) as usize,
                        language,
                        symbol_name,
                        symbol_kind,
                        signature_fragment,
                        visibility,
                        arity,
                        doc_comment_proximity,
                        content_hash,
                    },
                    distance,
                });
            }
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::VectorDb;
    use crate::parser::{content_hash_for_text, Chunk};
    use anyhow::Result;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{}-{}", prefix, nanos));
        fs::create_dir_all(&dir).expect("failed to create temporary test directory");
        dir
    }

    fn sample_chunk(file_path: &str, content: &str, start_line: usize, end_line: usize) -> Chunk {
        Chunk {
            file_path: file_path.to_string(),
            content: content.to_string(),
            start_line,
            end_line,
            language: "rust".to_string(),
            symbol_name: Some("sample_symbol".to_string()),
            symbol_kind: Some("function".to_string()),
            signature_fragment: None,
            visibility: None,
            arity: None,
            doc_comment_proximity: None,
            content_hash: content_hash_for_text(content),
        }
    }

    #[tokio::test]
    async fn replace_file_chunks_replaces_stale_rows_for_same_file() -> Result<()> {
        let db_dir = unique_temp_dir("zseek-db-upsert");
        let db = VectorDb::new(db_dir.to_string_lossy().as_ref()).await?;
        let file_path = "/tmp/project/src/auth.rs";

        let old_chunk = sample_chunk(file_path, "fn old_auth() {}", 1, 5);
        db.add_chunks(vec![(old_chunk, vec![0.1; 1536])]).await?;

        let new_chunk = sample_chunk(file_path, "fn new_auth() {}", 1, 5);
        let expected_hash = new_chunk.content_hash.clone();
        db.replace_file_chunks(file_path, vec![(new_chunk, vec![0.2; 1536])])
            .await?;

        let results = db.search(vec![0.2; 1536], 20).await?;
        let same_file = results
            .iter()
            .filter(|result| result.chunk.file_path == file_path)
            .collect::<Vec<_>>();

        assert_eq!(same_file.len(), 1);
        assert_eq!(same_file[0].chunk.content_hash, expected_hash);

        Ok(())
    }

    #[tokio::test]
    async fn replace_file_chunks_with_empty_payload_clears_existing_rows() -> Result<()> {
        let db_dir = unique_temp_dir("zseek-db-clear");
        let db = VectorDb::new(db_dir.to_string_lossy().as_ref()).await?;
        let file_path = "/tmp/project/src/stale.rs";

        let chunk = sample_chunk(file_path, "fn stale() {}", 10, 14);
        db.add_chunks(vec![(chunk, vec![0.3; 1536])]).await?;

        db.replace_file_chunks(file_path, Vec::new()).await?;

        let results = db.search(vec![0.3; 1536], 20).await?;
        assert!(
            results
                .iter()
                .all(|result| result.chunk.file_path != file_path)
        );

        Ok(())
    }
}
