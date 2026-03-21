use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

pub fn install_to_vscode() -> Result<()> {
    // Determine the VS Code user settings path
    let home = dirs::home_dir().context("Could not find home directory")?;
    
    let settings_paths = vec![
        // Mac
        home.join("Library/Application Support/Code/User/settings.json"),
        // Linux
        home.join(".config/Code/User/settings.json"),
        // Windows (will use APPDATA in practice, but usually we just fallback to AppData\Roaming if needed,
        // here we'll do a simple fallback for Windows since this is a basic script)
        // %APPDATA%\Code\User\settings.json
    ];

    let mut settings_file = None;
    for path in settings_paths {
        if path.exists() {
            settings_file = Some(path);
            break;
        }
    }

    let settings_path = match settings_file {
        Some(p) => p,
        None => {
            // Check for Windows AppData
            if let Ok(appdata) = std::env::var("APPDATA") {
                let p = PathBuf::from(appdata).join("Code\\User\\settings.json");
                if p.exists() {
                    Some(p)
                } else {
                    None
                }
            } else {
                None
            }
        }.context("Could not find VS Code settings.json. Ensure VS Code is installed.")?
    };

    println!("Found VS Code settings at: {}", settings_path.display());

    let current_exe = std::env::current_exe()?
        .to_str()
        .context("Invalid characters in executable path")?
        .to_string();

    let settings_content = fs::read_to_string(&settings_path)?;
    let mut config: serde_json::Value = serde_json::from_str(&settings_content).unwrap_or_else(|_| serde_json::json!({}));

    // Ensure the github.copilot.chat.mcp.servers object exists
    let mcp_key = "github.copilot.chat.mcp.servers";
    
    if !config.is_object() {
        config = serde_json::json!({});
    }

    let config_obj = config.as_object_mut().unwrap();
    if !config_obj.contains_key(mcp_key) {
        config_obj.insert(mcp_key.to_string(), serde_json::json!({}));
    }

    let mcp_servers = config_obj.get_mut(mcp_key).unwrap().as_object_mut().unwrap();

    // Insert Z-Seeker
    mcp_servers.insert(
        "Z-Seeker Search".to_string(),
        serde_json::json!({
            "command": current_exe,
            "args": []
        })
    );

    // Save back to settings.json
    let new_content = serde_json::to_string_pretty(&config)?;
    fs::write(&settings_path, new_content)?;

    println!("✅ Z-Seeker successfully installed to VS Code! Please 'Reload Window' in VS Code.");
    Ok(())
}
