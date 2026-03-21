use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct CopilotProvider {
    client: Client,
    token: String,
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
    pub fn new(token: String) -> Self {
        Self {
            client: Client::new(),
            token,
        }
    }

    pub async fn get_embeddings(&self, text: &str) -> Result<Vec<f32>> {
        if self.token.is_empty() {
            // Mock embedding for local testing if token is omitted
            tracing::warn!("No COPILOT_API_KEY set, generating dummy embedding (1536 dims)");
            return Ok(vec![0.01; 1536]);
        }

        let req_body = EmbeddingsRequest {
            input: text.to_string(),
            model: "text-embedding-3-small".to_string(), // Typical dense model size 1536
        };

        // This utilizes actual Copilot-compatible OpenAI endpoint or typical proxy pattern
        // In practice you hit the GitHub Copilot proxy `https://api.githubcopilot.com/embeddings`
        let resp = self.client
            .post("https://api.githubcopilot.com/embeddings")
            .bearer_auth(&self.token)
            .json(&req_body)
            .send()
            .await?
            .json::<EmbeddingsResponse>()
            .await?;

        resp.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("No embeddings returned from Copilot API")
    }
}
