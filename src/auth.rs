use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;

// We use the GitHub Copilot Plugin Client ID so it doesn't frighten the user with "repo" read/write scopes.
const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98"; 

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: u64,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    error: Option<String>,
}

pub fn get_token_path() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".copilot-mcp-token")
}

pub fn load_saved_token() -> Result<String> {
    let path = get_token_path();
    if path.exists() {
        let token = fs::read_to_string(path)?.trim().to_string();
        Ok(token)
    } else {
        Err(anyhow!("No saved token found"))
    }
}

pub async fn run_auth_flow() -> Result<()> {
    let client = Client::new();

    // 1. Request device code
    println!("Requesting device authorization from GitHub...");
    let res = client
        .post("https://github.com/login/device/code")
        .header("Accept", "application/json")
        .query(&[("client_id", CLIENT_ID), ("scope", "read:user")])
        .send()
        .await?;

    if !res.status().is_success() {
        return Err(anyhow!("Failed to request device code: {}", res.status()));
    }

    let device_auth: DeviceCodeResponse = res.json().await?;

    println!("\n=======================================================");
    println!("Please open the following URL in your browser:");
    println!("👉  {}", device_auth.verification_uri);
    println!("\nAnd enter the following code:");
    println!("🔑  {}", device_auth.user_code);
    println!("=======================================================\n");
    println!("Waiting for authorization...");

    // 2. Poll for token
    let mut interval = device_auth.interval;
    loop {
        sleep(Duration::from_secs(interval)).await;

        let token_res = client
            .post("https://github.com/login/oauth/access_token")
            .header("Accept", "application/json")
            .query(&[
                ("client_id", CLIENT_ID),
                ("device_code", device_auth.device_code.as_str()),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ])
            .send()
            .await?;

        let token_data: TokenResponse = token_res.json().await?;

        if let Some(token) = token_data.access_token {
            println!("✅ Successfully authenticated!");
            let path = get_token_path();
            fs::write(&path, token)?;
            println!("Token saved securely to: {}", path.display());
            break;
        }

        if let Some(err) = token_data.error {
            match err.as_str() {
                "authorization_pending" => {
                    // Still waiting, continue polling
                }
                "slow_down" => {
                    interval += 5;
                }
                "expired_token" => {
                    return Err(anyhow!("The device code has expired. Please run auth again."));
                }
                "access_denied" => {
                    return Err(anyhow!("Authorization was denied."));
                }
                _ => {
                    return Err(anyhow!("OAuth error: {}", err));
                }
            }
        }
    }

    Ok(())
}
