use anyhow::{anyhow, Context, Result};
use futures::stream::{self, StreamExt};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

const EMBEDDING_MODEL: &str = "text-embedding-3-small";
const EMBEDDING_DIMENSIONS: usize = 1536;
const EMBEDDING_BATCH_SIZE: usize = 24;
const EMBEDDING_MAX_CONCURRENCY: usize = 3;
const EMBEDDING_MAX_RETRIES: usize = 3;
const RETRY_BACKOFF_BASE_MS: u64 = 150;
const RETRY_BACKOFF_CAP_MS: u64 = 3000;

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
    pub input: Vec<String>,
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

#[derive(Debug)]
struct EmbeddingRequestFailure {
    message: String,
    retryable: bool,
}

impl EmbeddingRequestFailure {
    fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }

    fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }
}

fn batch_ranges(total_items: usize, batch_size: usize) -> Vec<(usize, usize)> {
    if total_items == 0 {
        return Vec::new();
    }

    let size = batch_size.max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;

    while start < total_items {
        let end = (start + size).min(total_items);
        ranges.push((start, end));
        start = end;
    }

    ranges
}

fn backoff_delay_ms(attempt: usize) -> u64 {
    let growth = 1u64 << attempt.min(6);
    RETRY_BACKOFF_BASE_MS
        .saturating_mul(growth)
        .min(RETRY_BACKOFF_CAP_MS)
}

fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
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
        let resp = self
            .client
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
            return Err(anyhow::anyhow!(
                "Failed to exchange Copilot token {}: {}",
                status,
                err_text
            ));
        }

        let token_resp: CopilotInternalTokenResponse = resp.json().await?;

        let mut cache = self.session_token.write().await;
        *cache = Some(SessionToken {
            token: token_resp.token.clone(),
            expires_at: token_resp.expires_at,
        });

        Ok(token_resp.token)
    }

    async fn request_embeddings_once(
        &self,
        inputs: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingRequestFailure> {
        let session_token = self
            .get_valid_session_token()
            .await
            .map_err(|err| EmbeddingRequestFailure::retryable(err.to_string()))?;

        let req_body = EmbeddingsRequest {
            input: inputs.to_vec(),
            model: EMBEDDING_MODEL.to_string(),
        };

        let resp = self
            .client
            .post("https://api.individual.githubcopilot.com/embeddings")
            .header("Authorization", format!("Bearer {}", session_token))
            .header("Editor-Version", "vscode/1.85.0")
            .header("Editor-Plugin-Version", "copilot-chat/0.11.1")
            .header("Accept", "application/json")
            .header("User-Agent", "zseek")
            .json(&req_body)
            .send()
            .await
            .map_err(|err| {
                let retryable = err.is_timeout() || err.is_connect() || err.is_request();
                let message = format!("Embedding request transport error: {}", err);
                if retryable {
                    EmbeddingRequestFailure::retryable(message)
                } else {
                    EmbeddingRequestFailure::fatal(message)
                }
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();

            // If the session token is rejected, force flush the cache so future attempts refresh it.
            if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
                let mut cache = self.session_token.write().await;
                *cache = None;
            }

            let message = format!("GitHub API Error {}: {}", status, err_text);
            let retryable = is_retryable_status(status)
                || status == StatusCode::UNAUTHORIZED
                || status == StatusCode::FORBIDDEN;

            return if retryable {
                Err(EmbeddingRequestFailure::retryable(message))
            } else {
                Err(EmbeddingRequestFailure::fatal(message))
            };
        }

        let resp_json = resp
            .json::<EmbeddingsResponse>()
            .await
            .map_err(|err| EmbeddingRequestFailure::fatal(format!("Invalid embeddings response: {}", err)))?;

        if resp_json.data.len() != inputs.len() {
            return Err(EmbeddingRequestFailure::fatal(format!(
                "Embedding count mismatch: requested {}, received {}",
                inputs.len(),
                resp_json.data.len()
            )));
        }

        Ok(resp_json.data.into_iter().map(|d| d.embedding).collect())
    }

    async fn request_embeddings_with_retry(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut attempt = 0usize;

        loop {
            match self.request_embeddings_once(inputs).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(err) if err.retryable && attempt < EMBEDDING_MAX_RETRIES => {
                    let delay_ms = backoff_delay_ms(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        delay_ms,
                        "Retrying embedding batch after transient failure"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    attempt += 1;
                }
                Err(err) => return Err(anyhow!(err.message)),
            }
        }
    }

    pub async fn get_embeddings_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if self.oauth_token.is_empty() {
            tracing::warn!(
                "No COPILOT_API_KEY set, generating dummy embeddings ({} dims)",
                EMBEDDING_DIMENSIONS
            );
            return Ok(vec![vec![0.01; EMBEDDING_DIMENSIONS]; texts.len()]);
        }

        let ranges = batch_ranges(texts.len(), EMBEDDING_BATCH_SIZE);
        let batch_count = ranges.len();

        let indexed_results = stream::iter(ranges.into_iter().enumerate())
            .map(|(batch_idx, (start, end))| {
                let payload = texts[start..end].to_vec();
                async move {
                    let embeddings = self.request_embeddings_with_retry(&payload).await?;
                    Ok::<(usize, Vec<Vec<f32>>), anyhow::Error>((batch_idx, embeddings))
                }
            })
            .buffer_unordered(EMBEDDING_MAX_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;

        let mut ordered_batches = vec![None; batch_count];
        for entry in indexed_results {
            let (batch_idx, embeddings) = entry?;
            ordered_batches[batch_idx] = Some(embeddings);
        }

        let mut flattened = Vec::with_capacity(texts.len());
        for maybe_batch in ordered_batches {
            let batch = maybe_batch.context("Missing embedding batch result")?;
            flattened.extend(batch);
        }

        if flattened.len() != texts.len() {
            return Err(anyhow!(
                "Embedding count mismatch after batching: requested {}, received {}",
                texts.len(),
                flattened.len()
            ));
        }

        Ok(flattened)
    }

    pub async fn get_embeddings(&self, text: &str) -> Result<Vec<f32>> {
        let inputs = vec![text.to_string()];
        let mut results = self.get_embeddings_batch(&inputs).await?;
        results
            .pop()
            .context("No embeddings returned from Copilot API")
    }
}

#[cfg(test)]
mod tests {
    use super::{backoff_delay_ms, batch_ranges, is_retryable_status};
    use reqwest::StatusCode;

    #[test]
    fn batch_ranges_covers_all_items_in_order() {
        let ranges = batch_ranges(10, 4);
        assert_eq!(ranges, vec![(0, 4), (4, 8), (8, 10)]);
    }

    #[test]
    fn batch_ranges_handles_small_or_empty_inputs() {
        assert_eq!(batch_ranges(0, 4), Vec::<(usize, usize)>::new());
        assert_eq!(batch_ranges(2, 8), vec![(0, 2)]);
    }

    #[test]
    fn backoff_delay_grows_monotonically_with_attempts() {
        let first = backoff_delay_ms(0);
        let second = backoff_delay_ms(1);
        let third = backoff_delay_ms(2);

        assert!(second > first);
        assert!(third > second);
    }

    #[test]
    fn retryable_status_policy_matches_transient_errors() {
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(is_retryable_status(StatusCode::SERVICE_UNAVAILABLE));

        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(StatusCode::UNAUTHORIZED));
    }
}
