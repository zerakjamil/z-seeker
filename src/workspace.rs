use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct WorkspaceFile {
    #[serde(default)]
    folders: Vec<WorkspaceFolder>,
}

#[derive(Debug, Deserialize)]
struct WorkspaceFolder {
    path: Option<String>,
    uri: Option<String>,
}

pub fn resolve_scope_roots(current_dir: &Path, workspace_file: Option<&Path>) -> Result<Vec<PathBuf>> {
    if let Some(workspace_file) = workspace_file {
        let workspace_file_path = if workspace_file.is_absolute() {
            workspace_file.to_path_buf()
        } else {
            current_dir.join(workspace_file)
        };

        return resolve_workspace_file_roots(&workspace_file_path);
    }

    let root = current_dir
        .canonicalize()
        .unwrap_or_else(|_| normalize_path_lexically(current_dir));
    Ok(vec![root])
}

fn resolve_workspace_file_roots(workspace_file: &Path) -> Result<Vec<PathBuf>> {
    let raw = std::fs::read_to_string(workspace_file).with_context(|| {
        format!(
            "Failed reading workspace file at {}",
            workspace_file.display()
        )
    })?;

    let parsed = serde_json::from_str::<WorkspaceFile>(&raw).with_context(|| {
        format!(
            "Failed parsing workspace JSON at {}",
            workspace_file.display()
        )
    })?;

    if parsed.folders.is_empty() {
        return Err(anyhow!(
            "Workspace file {} has no folders",
            workspace_file.display()
        ));
    }

    let base_dir = workspace_file.parent().unwrap_or_else(|| Path::new("."));
    let mut roots = Vec::new();
    let mut seen = HashSet::new();

    for folder in parsed.folders {
        let raw_value = folder.path.as_deref().or(folder.uri.as_deref());
        let Some(raw_value) = raw_value else {
            continue;
        };

        let Some(folder_path) = parse_path_or_uri(raw_value) else {
            continue;
        };

        let resolved = if folder_path.is_absolute() {
            folder_path
        } else {
            base_dir.join(folder_path)
        };

        let canonical = resolved
            .canonicalize()
            .unwrap_or_else(|_| normalize_path_lexically(&resolved));

        if !canonical.is_dir() {
            continue;
        }

        if seen.insert(canonical.clone()) {
            roots.push(canonical);
        }
    }

    if roots.is_empty() {
        return Err(anyhow!(
            "Workspace file {} contains no valid folders",
            workspace_file.display()
        ));
    }

    Ok(roots)
}

fn parse_path_or_uri(raw: &str) -> Option<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(uri_path) = trimmed.strip_prefix("file://") {
        #[cfg(windows)]
        {
            let mut decoded = uri_path.replace("%20", " ");
            if decoded.starts_with('/')
                && decoded.len() > 2
                && decoded.as_bytes().get(2) == Some(&b':')
            {
                decoded.remove(0);
            }
            return Some(PathBuf::from(decoded));
        }

        #[cfg(not(windows))]
        {
            return Some(PathBuf::from(uri_path.replace("%20", " ")));
        }
    }

    if trimmed.contains("://") {
        return None;
    }

    Some(PathBuf::from(trimmed))
}

fn normalize_path_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::resolve_scope_roots;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("{}-{}-{}", prefix, std::process::id(), suffix))
    }

    fn create_workspace_file(path: &Path, content: &str) {
        fs::write(path, content).expect("write workspace file");
    }

    #[test]
    fn defaults_to_current_dir_when_workspace_file_is_missing() {
        let root = unique_temp_dir("zseek-default-root");
        fs::create_dir_all(&root).expect("create root");

        let roots = resolve_scope_roots(&root, None).expect("resolve default root");
        let expected = root.canonicalize().expect("canonicalize root");

        assert_eq!(roots, vec![expected]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolves_relative_workspace_folders() {
        let base = unique_temp_dir("zseek-workspace-relative");
        let project_a = base.join("project-a");
        let project_b = base.join("project-b");
        fs::create_dir_all(&project_a).expect("create project a");
        fs::create_dir_all(&project_b).expect("create project b");

        let workspace_file = base.join("multi.code-workspace");
        create_workspace_file(
            &workspace_file,
            r#"{
                "folders": [
                    {"path": "project-a"},
                    {"path": "project-b"}
                ]
            }"#,
        );

        let roots = resolve_scope_roots(&base, Some(&workspace_file)).expect("resolve roots");
        let expected_a = project_a.canonicalize().expect("canonicalize project a");
        let expected_b = project_b.canonicalize().expect("canonicalize project b");

        assert_eq!(roots.len(), 2);
        assert!(roots.iter().any(|root| root == &expected_a));
        assert!(roots.iter().any(|root| root == &expected_b));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resolves_workspace_folder_file_uri() {
        let base = unique_temp_dir("zseek-workspace-uri");
        let project = base.join("project with spaces");
        fs::create_dir_all(&project).expect("create project");

        let workspace_file = base.join("uri.code-workspace");
        let encoded_uri = format!(
            "file://{}",
            project
                .to_string_lossy()
                .replace(' ', "%20")
        );

        create_workspace_file(
            &workspace_file,
            &format!(
                "{{\n  \"folders\": [{{\"uri\": \"{}\"}}]\n}}",
                encoded_uri
            ),
        );

        let roots = resolve_scope_roots(&base, Some(&workspace_file)).expect("resolve uri roots");
        let expected = project.canonicalize().expect("canonicalize project");

        assert_eq!(roots, vec![expected]);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn errors_when_workspace_has_no_valid_folders() {
        let base = unique_temp_dir("zseek-workspace-invalid");
        fs::create_dir_all(&base).expect("create base");

        let workspace_file = base.join("invalid.code-workspace");
        create_workspace_file(
            &workspace_file,
            r#"{
                "folders": [
                    {"path": "missing-folder"},
                    {"uri": "https://example.com/not-local"}
                ]
            }"#,
        );

        let err = resolve_scope_roots(&base, Some(&workspace_file)).expect_err("expected error");
        assert!(
            err.to_string().contains("contains no valid folders"),
            "unexpected error: {err}"
        );

        let _ = fs::remove_dir_all(&base);
    }
}
