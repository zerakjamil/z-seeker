use anyhow::Result;
use std::sync::Arc;
#[tokio::main]
async fn main() -> Result<()> {
    let uri = "/Users/zerakjamil/ChessTacticsOffline/.lancedb";
    let conn = lancedb::connect(uri).execute().await.unwrap();
    let table = conn.open_table("semantic_chunks").execute().await.unwrap();
    let count = table.count_rows(None).await.unwrap();
    println!("Rows in DB: {}", count);
    Ok(())
}
