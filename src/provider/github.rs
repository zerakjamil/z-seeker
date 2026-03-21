use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct CopilotProvider {
    client: Client,
    oauth_token: String,
    session_token: Arc<RwLock<Option<SessionToken>>>,
}

#[derive(Clone, Debug)]
struct SessionToken {
    token: String,
    expires_at: u64,
}

#[derive(Deserialize, Debug)]
struct CopilotInternalTokenResponse {
    token: String,
    expires_at: u64,
    // endpoints: std::collections::HashMap<String, String>, // Unused for now
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingsRequest {
    pub input: String,
    pub model: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingsResponse {
    pub data: Vec<EmbeddingData>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
}

impl CopilotProvider {
    pub fn new(oauth_token: String) -> Self {
        Self {
            client: Client::new(),
            oauth_token,
            session_token: Arc::new(RwLock::new(None)),
        }
    }

    async fn get_valid_session_token(&self) -> Result<String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Check if we have a valid cached token
        {
            let cache = self.session_token.read().await;
            if let Some(session) = cache.as_ref() {
                // Buffer of 60 seconds to mitigate edge case expiration
                if session.expires_at > now + 60 {
                    return Ok(session.token.clone());
                }
            }
        }

        // Token is missing or expired, fetch a new one
        let resp = self.client
            .get("https://api.github.com/copilot_internal/v2/token")
            .header("Authorization", format!("token {}", self.oauth_token))
            .header("Editor-Version", "vscode/1.85.0")
            .header("Editor-Plugin-Version", "copilot-chat/0.11.1")
            .header("Accept", "application/json")
            .header("User-Agent", "zseek")
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to exchange Copilot token {}: {}", status, err_text));
        }

        let token_resp: CopilotInternalTokenResponse = resp.json().await?;

        let mut cache = self.session_token.write().await;
        *cache = Some(SessionToken {
            token: token_resp.token.clone(),
            expires_at: token_resp.expires_at,
        });

        Ok(token_resp.token)
    }

    pub async fn get_embeddings(&self, text: &str) -> Result<Vec<f32>> {
        if self.oauth_token.is_empty() {
            // Mock embedding for local testing if token is omitted
            tracing::warn!("No COPILOT_API_KEY set, generating dummy embedding (1536 dims)");
            return Ok(vec![0.01; 1536]);
        }

        let session_token = self.get_valid_session_token().await?;

        let req_body = EmbeddingsRequest {
            input: text.to_string(),
            model: "text-embedding-3-small".to_string(), // Typical dense model size 1536
        };

        // Notice we use the standard Copilot telemetry proxy which requires Editor-Version
        let resp = self.client
            .post("https://api.individual.githubcopilot.com/embeddings")
            .header("Authorization", format!("Bearer {}", session_token))
            .header("Editor-Version", "vscode/1.85.0")
            .header("Editor-Plugin-Version", "copilot-chat/0.11.1")
            .header("Accept", "application/json")
            .header("User-Agent", "zseek")
            .json(&req_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();
            
            // If the session token is rejected, force flush the cache so it retries next time
            if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
                let mut cache = self.session_token.write().await;
                *cache = None;
            }

            return Err(anyhow::anyhow!("GitHub API Error {}: {}", status, err_text));
        }

        let resp_json = resp.json::<EmbeddingsResponse>().await?;

        resp_json.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("No embeddings returned from Copilot API")
    }
}
