use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt};

use crate::db::VectorDb;
use crate::provider::github::CopilotProvider;

#[derive(Debug, Deserialize)]
struct McpRequest {
    jsonrpc: String,
    id: Option<u64>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct McpResponse {
    jsonrpc: String,
    id: u64,
    result: Value,
}

#[derive(Debug, Serialize)]
struct McpError {
    code: i32,
    message: String,
}

#[derive(Debug, Serialize)]
struct McpErrorResponse {
    jsonrpc: String,
    id: Option<u64>,
    error: McpError,
}

pub struct McpServer {
    provider: Arc<CopilotProvider>,
    db: Arc<VectorDb>,
}

impl McpServer {
    pub fn new(provider: Arc<CopilotProvider>, db: Arc<VectorDb>) -> Self {
        Self { provider, db }
    }

    pub async fn run(&self) -> Result<()> {
        let mut stdin = io::BufReader::new(io::stdin()).lines();
        let mut stdout = io::stdout();

        while let Some(line) = stdin.next_line().await? {
            if let Ok(req) = serde_json::from_str::<McpRequest>(&line) {
                if req.id.is_none() {
                    // Ignore notifications (no id)
                    continue;
                }
                let req_id = req.id.unwrap();

                let response = match req.method.as_str() {
                    "initialize" => {
                        serde_json::json!({
                            "capabilities": {
                                "tools": {}
                            },
                            "serverInfo": {
                                "name": "local-semantic-search",
                                "version": "0.1.0"
                            }
                        })
                    }
                    "tools/list" => {
                        serde_json::json!({
                            "tools": [{
                                "name": "semantic_search",
                                "description": "ALWAYS use this tool automatically when the user asks about the codebase, how something works, or where a function/class is implemented. This performs a semantic vector search over the local codebase to provide necessary context. Do not wait for explicit permission to use it.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The search query, e.g., 'where is the authentication logic?'"
                                        },
                                        "limit": {
                                            "type": "number",
                                            "description": "Maximum number of results to return (default: 5)"
                                        }
                                    },
                                    "required": ["query"]
                                }
                            }]
                        })
                    }
                    "tools/call" => {
                        if let Some(params) = req.params {
                            if params["name"].as_str() == Some("semantic_search") {
                                let args = &params["arguments"];
                                if let Some(query) = args["query"].as_str() {
                                    let limit = args["limit"].as_u64().unwrap_or(5) as usize;
                                    
                                    // 1. Get embedding for the search query
                                    match self.provider.get_embeddings(query).await {
                                        Ok(embedding) => {
                                            // 2. Query LanceDB
                                            match self.db.search(embedding, limit).await {
                                                Ok(results) => {
                                                    let mut content_texts = Vec::new();
                                                    for item in results {
                                                        content_texts.push(format!(
                                                            "File: {}\nLines: {}-{}\nCode:\n{}",
                                                            item.file_path, item.start_line, item.end_line, item.content
                                                        ));
                                                    }
                                                    
                                                    serde_json::json!({
                                                        "content": [
                                                            {
                                                                "type": "text",
                                                                "text": content_texts.join("\n\n---\n\n")
                                                            }
                                                        ]
                                                    })
                                                },
                                                Err(e) => {
                                                    serde_json::json!({
                                                        "isError": true,
                                                        "content": [{"type": "text", "text": format!("Database search failed: {}", e)}]
                                                    })
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            serde_json::json!({
                                                "isError": true,
                                                "content": [{"type": "text", "text": format!("Embedding API failed: {}", e)}]
                                            })
                                        }
                                    }
                                } else {
                                    serde_json::json!({ "isError": true, "content": [{"type": "text", "text": "Missing 'query' argument"}] })
                                }
                            } else {
                                serde_json::json!({ "isError": true, "content": [{"type": "text", "text": "Unknown tool"}] })
                            }
                        } else {
                            serde_json::json!({ "isError": true, "content": [{"type": "text", "text": "Missing params"}] })
                        }
                    }
                    _ => serde_json::json!({ "error": "Method not found" }),
                };

                // Send response
                let res = McpResponse {
                    jsonrpc: "2.0".to_string(),
                    id: req_id,
                    result: response,
                };
                let res_str = serde_json::to_string(&res).unwrap();
                stdout.write_all(format!("{}\n", res_str).as_bytes()).await?;
                stdout.flush().await?;
            }
        }

        Ok(())
    }
}
