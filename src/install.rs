use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

const SELF_UPDATE_REPO: &str = "https://github.com/zerakjamil/z-seeker";

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
        }
        .context("Could not find VS Code settings.json. Ensure VS Code is installed.")?,
    };

    println!("Found VS Code settings at: {}", settings_path.display());

    let current_exe = std::env::current_exe()?
        .to_str()
        .context("Invalid characters in executable path")?
        .to_string();

    // We should install to BOTH settings.json and the new mcp.json standard

    let mcp_json_path = settings_path.parent().unwrap().join("mcp.json");
    if mcp_json_path.exists() {
        let mcp_content = fs::read_to_string(&mcp_json_path).unwrap_or_else(|_| "{}".to_string());
        let mut mcp_config: serde_json::Value =
            serde_json::from_str(&mcp_content).unwrap_or_else(|_| serde_json::json!({}));

        let mcp_obj = mcp_config.as_object_mut().unwrap();
        if !mcp_obj.contains_key("servers") {
            mcp_obj.insert("servers".to_string(), serde_json::json!({}));
        }

        let servers = mcp_obj.get_mut("servers").unwrap().as_object_mut().unwrap();
        servers.insert(
            "Z-Seeker".to_string(),
            serde_json::json!({
                "command": &current_exe,
                "args": []
            }),
        );
        fs::write(&mcp_json_path, serde_json::to_string_pretty(&mcp_config)?)?;
        println!(
            "✅ Injected into global mcp.json at: {}",
            mcp_json_path.display()
        );
    } else {
        // Create mcp.json if it doesn't exist
        let mcp_config = serde_json::json!({
            "servers": {
                "Z-Seeker": {
                    "command": &current_exe,
                    "args": []
                }
            }
        });
        fs::write(&mcp_json_path, serde_json::to_string_pretty(&mcp_config)?)?;
        println!(
            "✅ Created and injected into mcp.json at: {}",
            mcp_json_path.display()
        );
    }

    // Now do the Copilot settings.json
    let settings_content = fs::read_to_string(&settings_path)?;
    let mut config: serde_json::Value =
        serde_json::from_str(&settings_content).unwrap_or_else(|_| serde_json::json!({}));

    // Ensure the github.copilot.chat.mcp.servers object exists
    let mcp_key = "github.copilot.chat.mcp.servers";

    if !config.is_object() {
        config = serde_json::json!({});
    }

    let config_obj = config.as_object_mut().unwrap();
    if !config_obj.contains_key(mcp_key) {
        config_obj.insert(mcp_key.to_string(), serde_json::json!({}));
    }

    let mcp_servers = config_obj
        .get_mut(mcp_key)
        .unwrap()
        .as_object_mut()
        .unwrap();

    // Insert Z-Seeker
    mcp_servers.insert(
        "Z-Seeker Search".to_string(),
        serde_json::json!({
            "command": current_exe,
            "args": []
        }),
    );

    // Save back to settings.json
    let new_content = serde_json::to_string_pretty(&config)?;
    fs::write(&settings_path, new_content)?;

    println!("✅ Z-Seeker successfully installed to VS Code! Please 'Reload Window' in VS Code.");
    Ok(())
}

fn infer_install_root(current_exe: &Path) -> Option<PathBuf> {
    let bin_dir = current_exe.parent()?;
    if bin_dir.file_name().and_then(|name| name.to_str()) == Some("bin") {
        return bin_dir.parent().map(Path::to_path_buf);
    }

    None
}

fn self_update_command_args(current_exe: &Path) -> Vec<String> {
    let mut args = vec![
        "install".to_string(),
        "--git".to_string(),
        SELF_UPDATE_REPO.to_string(),
        "--bin".to_string(),
        "zseek".to_string(),
        "--force".to_string(),
        "--locked".to_string(),
    ];

    if let Some(root) = infer_install_root(current_exe) {
        args.push("--root".to_string());
        args.push(root.to_string_lossy().to_string());
    }

    args
}

pub fn self_update() -> Result<()> {
    let current_exe = std::env::current_exe().context("Failed to resolve current executable path")?;
    let cargo_bin = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let args = self_update_command_args(&current_exe);

    println!("Updating zseek to the newest version from {}...", SELF_UPDATE_REPO);

    let status = std::process::Command::new(&cargo_bin)
        .args(&args)
        .status()
        .with_context(|| format!("Failed to execute '{}' for self-update", cargo_bin))?;

    if !status.success() {
        return Err(anyhow!(
            "Self-update failed. Please re-run with logs: {} {}",
            cargo_bin,
            args.join(" ")
        ));
    }

    println!("✅ zseek updated successfully.");
    println!("Installed executable path: {}", current_exe.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{infer_install_root, self_update_command_args, SELF_UPDATE_REPO};
    use std::path::{Path, PathBuf};

    #[test]
    fn infer_install_root_detects_bin_layout() {
        let root = infer_install_root(Path::new("/home/aywar/.local/bin/zseek"));
        assert_eq!(root, Some(PathBuf::from("/home/aywar/.local")));
    }

    #[test]
    fn infer_install_root_returns_none_outside_bin_layout() {
        let root = infer_install_root(Path::new("/opt/zseek/zseek"));
        assert!(root.is_none());
    }

    #[test]
    fn self_update_args_include_repo_and_root_when_available() {
        let args = self_update_command_args(Path::new("/home/aywar/.local/bin/zseek"));

        assert!(args.contains(&"--git".to_string()));
        assert!(args.contains(&SELF_UPDATE_REPO.to_string()));
        assert!(args.contains(&"--root".to_string()));
        assert!(args.contains(&"/home/aywar/.local".to_string()));
    }

    #[test]
    fn self_update_args_skip_root_when_layout_is_unknown() {
        let args = self_update_command_args(Path::new("/opt/zseek/zseek"));
        assert!(!args.contains(&"--root".to_string()));
    }
}
