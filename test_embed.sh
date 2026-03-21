curl -s -w "\nHTTP_STATUS: %{http_code}" -X POST https://api.githubcopilot.com/embeddings \
  -H "Authorization: Bearer $(cat ~/.copilot-mcp-token)" \
  -H "Editor-Version: vscode/1.85.0" \
  -H "Editor-Plugin-Version: copilot-chat/0.11.1" \
  -H "Content-Type: application/json" \
  -d '{"input":"test", "model":"text-embedding-3-small"}'
