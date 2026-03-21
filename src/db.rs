use std::sync::Arc;
use anyhow::{Context, Result};
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, StringArray, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lancedb::connection::Connection;
use lancedb::{connect, Table};
use tracing::info;
use futures::StreamExt;
use crate::parser::Chunk;

use lancedb::query::{ExecutableQuery, QueryBase};
pub struct VectorDb {
    table: Table,
}

impl VectorDb {
    pub async fn new(uri: &str) -> Result<Self> {
        let conn = connect(uri).execute().await?;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("start_line", DataType::Int32, false),
            Field::new("end_line", DataType::Int32, false),
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
                conn.create_empty_table(table_name, schema).execute().await?
            }
        };

        Ok(Self { table })
    }

    pub async fn add_chunks(&self, data: Vec<(Chunk, Vec<f32>)>) -> Result<()> {
        if data.is_empty() { return Ok(()); }
        let schema = self.table.schema().await?;

        let mut file_paths = Vec::with_capacity(data.len());
        let mut contents = Vec::with_capacity(data.len());
        let mut start_lines = Vec::with_capacity(data.len());
        let mut end_lines = Vec::with_capacity(data.len());
        let mut vectors = Vec::with_capacity(data.len());

        for (chunk, embedding) in data {
            file_paths.push(chunk.file_path);
            contents.push(chunk.content);
            start_lines.push(chunk.start_line as i32);
            end_lines.push(chunk.end_line as i32);
            vectors.push(Some(embedding.into_iter().map(Some).collect::<Vec<_>>()));
        }

        let file_path_array = StringArray::from(file_paths);
        let content_array = StringArray::from(contents);
        let start_line_array = Int32Array::from(start_lines);
        let end_line_array = Int32Array::from(end_lines);
        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, 1536);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
            ],
        )?;

        let batches = vec![Ok(batch)];
        let iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());
        self.table.add(iter).execute().await?;
        Ok(())
    }

    pub async fn add_chunk(&self, chunk: Chunk, embedding: Vec<f32>) -> Result<()> {
        let schema = self.table.schema().await?;

        let file_path_array = StringArray::from(vec![chunk.file_path.clone()]);
        let content_array = StringArray::from(vec![chunk.content.clone()]);
        let start_line_array = Int32Array::from(vec![chunk.start_line as i32]);
        let end_line_array = Int32Array::from(vec![chunk.end_line as i32]);

        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(
                embedding.into_iter().map(Some).collect::<Vec<_>>(),
            )],
            1536,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(file_path_array) as Arc<dyn arrow_array::Array>,
                Arc::new(content_array) as Arc<dyn arrow_array::Array>,
                Arc::new(start_line_array) as Arc<dyn arrow_array::Array>,
                Arc::new(end_line_array) as Arc<dyn arrow_array::Array>,
                Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
            ],
        )?;

        // LanceDB 0.14 IntoArrow is implemented for RecordBatchIterator directly
        let batches = vec![Ok(batch)];
        let iter = RecordBatchIterator::new(batches.into_iter(), schema.clone());
        self.table.add(iter).execute().await?;
        Ok(())
    }

    pub async fn search(&self, query_embedding: Vec<f32>, limit_count: usize) -> Result<Vec<Chunk>> {
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
            if batch.num_rows() == 0 { continue; }

            let file_paths = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let contents = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let start_lines = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
            let end_lines = batch.column(3).as_any().downcast_ref::<Int32Array>().unwrap();

            for i in 0..batch.num_rows() {
                chunks.push(Chunk {
                    file_path: file_paths.value(i).to_string(),
                    content: contents.value(i).to_string(),
                    start_line: start_lines.value(i) as usize,
                    end_line: end_lines.value(i) as usize,
                });
            }
        }
        Ok(chunks)
    }
}
