use std::sync::Arc;
#[tokio::main]
async fn main() {
    let db = lancedb::connect("sqlite://").execute().await.unwrap();
    // This is just a test to see if LanceDB fails when printing.
    // Actually let's just inspect the .lancedb folder
}
