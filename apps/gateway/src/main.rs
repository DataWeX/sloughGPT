use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::{get, post, any},
    Json, Router,
};
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct GatewayConfig {
    python_core_url: String,
    static_dir: String,
    listen_addr: SocketAddr,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        let port = std::env::var("MAN_GATEWAY_PORT")
            .unwrap_or_else(|_| "8080".into())
            .parse()
            .unwrap_or(8080);
        Self {
            python_core_url: std::env::var("MAN_CORE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8000".into()),
            static_dir: std::env::var("MAN_STATIC_DIR")
                .unwrap_or_else(|_| "apps/web/.next/static".into()),
            listen_addr: SocketAddr::from(([0, 0, 0, 0], port)),
        }
    }
}

// ── Shared State ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    config: GatewayConfig,
    http: Client,
    core_status: Arc<RwLock<CoreStatus>>,
}

#[derive(Clone, Default, Serialize)]
struct CoreStatus {
    healthy: bool,
    last_check: Option<chrono::DateTime<chrono::Utc>>,
    model_loaded: bool,
    model_name: String,
}

// ── Type-Safe Request/Response Models ───────────────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_max_tokens() -> u32 { 512 }
fn default_temperature() -> f32 { 0.8 }

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    message: String,
    session_id: String,
    done: bool,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_new_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
    model: String,
    tokens_generated: u32,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    gateway: String,
    sidecar: CoreStatus,
    uptime_seconds: u64,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    name: String,
    loaded: bool,
    source: String,
}

// ── Errors ──────────────────────────────────────────────────────────────────

struct GatewayError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}

impl From<reqwest::Error> for GatewayError {
    fn from(e: reqwest::Error) -> Self {
        GatewayError {
            status: StatusCode::BAD_GATEWAY,
            message: format!("Sidecar error: {}", e),
        }
    }
}

// ── Startup ─────────────────────────────────────────────────────────────────

static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "slough_gateway=info,tower_http=info".into()),
        )
        .init();

    let config = GatewayConfig::default();
    let _ = START_TIME.set(std::time::Instant::now());

    let http = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("Failed to create HTTP client");

    let state = AppState {
        config: config.clone(),
        http,
        core_status: Arc::new(RwLock::new(CoreStatus::default())),
    };

    // Spawn sidecar health checker
    let health_state = state.clone();
    tokio::spawn(async move {
        loop {
            check_sidecar_health(&health_state).await;
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    let app = Router::new()
        // Health & info
        .route("/health", get(health_check))
        .route("/health/detailed", get(detailed_health))

        // Chat (streaming SSE + non-streaming)
        .route("/chat", post(chat_handler))
        .route("/chat/stream", post(chat_stream_handler))

        // Generation
        .route("/inference/generate", post(generate_handler))
        .route("/inference/generate/stream", post(generate_stream_handler))

        // Models (proxy)
        .route("/models", get(proxy_get))
        .route("/models/hf", get(proxy_get))
        .route("/models/load", post(proxy_post))
        .route("/models/unload", post(proxy_post))

        // Training (proxy)
        .route("/training/start", post(proxy_post))
        .route("/training/jobs", get(proxy_get))
        .route("/auto-train/start", post(proxy_post))
        .route("/auto-train/stream", get(proxy_sse))

        // Knowledge (proxy)
        .route("/knowledge", get(proxy_get))
        .route("/knowledge", post(proxy_post))

        // Datasets (proxy)
        .route("/datasets", get(proxy_get))
        .route("/datasets", post(proxy_post))

        // Souls (proxy)
        .route("/souls", get(proxy_get))
        .route("/souls/current", get(proxy_get))
        .route("/souls/switch", post(proxy_post))

        // Sessions (proxy)
        .route("/session/{id}/context", post(proxy_post))
        .route("/session/{id}/regenerate", post(proxy_sse))

        // Feedback (proxy)
        .route("/feedback/workflow-record", post(proxy_post))

        // Catch-all proxy (must be last — uses fallback)
        .fallback(catch_all_proxy)

        .layer(CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .expect("Failed to bind");

    info!("🌐 Gateway listening on {}", config.listen_addr);
    info!("   → Python sidecar at {}", config.python_core_url);

    axum::serve(listener, app)
        .await
        .expect("Server failed");
}

// ── Health Handlers ─────────────────────────────────────────────────────────

async fn health_check(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    let sidecar = state.core_status.read().await.clone();
    let uptime = START_TIME.get()
        .map(|t| t.elapsed().as_secs())
        .unwrap_or(0);

    Json(HealthResponse {
        status: if sidecar.healthy { "ok".into() } else { "degraded".into() },
        gateway: "rust".into(),
        sidecar,
        uptime_seconds: uptime,
    })
}

async fn detailed_health(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let sidecar = state.core_status.read().await.clone();
    let uptime = START_TIME.get()
        .map(|t| t.elapsed().as_secs())
        .unwrap_or(0);

    Json(serde_json::json!({
        "gateway": {
            "status": "ok",
            "version": env!("CARGO_PKG_VERSION"),
            "uptime_seconds": uptime,
        },
        "sidecar": sidecar,
    }))
}

async fn check_sidecar_health(state: &AppState) {
    let url = format!("{}/health", state.config.python_core_url);
    match state.http.get(&url).send().await {
        Ok(resp) => {
            if let Ok(health) = resp.json::<serde_json::Value>().await {
                let mut status = state.core_status.write().await;
                status.healthy = true;
                status.last_check = Some(chrono::Utc::now());
                status.model_loaded = health.get("model_loaded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                status.model_name = health.get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
            }
        }
        Err(_) => {
            let mut status = state.core_status.write().await;
            status.healthy = false;
            status.last_check = Some(chrono::Utc::now());
        }
    }
}

// ── Chat Handlers ───────────────────────────────────────────────────────────

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, GatewayError> {
    let url = format!("{}/chat", state.config.python_core_url);
    let resp = state.http.post(&url)
        .json(&serde_json::json!({
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }))
        .send()
        .await?;

    let body: serde_json::Value = resp.json().await?;
    let message = body.get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(Json(ChatResponse {
        message,
        session_id: body.get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        done: true,
    }))
}

async fn chat_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
    let url = format!("{}/chat/stream", state.config.python_core_url);
    let http = state.http.clone();

    let stream = async_stream::stream! {
        let resp = http.post(&url)
            .json(&serde_json::json!({
                "messages": req.messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            }))
            .send()
            .await;

        let mut resp = match resp {
            Ok(r) => r,
            Err(e) => {
                let evt = axum::response::sse::Event::default()
                    .data(format!("{{\"error\":\"{}\"}}", e));
                yield Ok(evt);
                return;
            }
        };

        while let Some(chunk) = resp.chunk().await.unwrap_or(None) {
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    let evt = axum::response::sse::Event::default()
                        .data(data.to_string());
                    yield Ok(evt);
                }
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

// ── Generation Handlers ─────────────────────────────────────────────────────

async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, GatewayError> {
    let url = format!("{}/inference/generate", state.config.python_core_url);
    let resp = state.http.post(&url)
        .json(&serde_json::json!({
            "prompt": req.prompt,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
        }))
        .send()
        .await?;

    let body: serde_json::Value = resp.json().await?;
    Ok(Json(GenerateResponse {
        text: body.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        model: body.get("model").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
        tokens_generated: body.get("tokens_generated").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
    }))
}

async fn generate_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let url = format!("{}/inference/generate/stream", state.config.python_core_url);
    let http = state.http.clone();

    let stream = async_stream::stream! {
        let resp = http.post(&url)
            .json(&serde_json::json!({
                "prompt": req.prompt,
                "max_new_tokens": req.max_new_tokens,
                "temperature": req.temperature,
            }))
            .send()
            .await;

        let mut resp = match resp {
            Ok(r) => r,
            Err(e) => {
                let evt = axum::response::sse::Event::default()
                    .data(format!("{{\"error\":\"{}\"}}", e));
                yield Ok(evt);
                return;
            }
        };

        while let Some(chunk) = resp.chunk().await.unwrap_or(None) {
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    let evt = axum::response::sse::Event::default()
                        .data(data.to_string());
                    yield Ok(evt);
                }
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

// ── Generic Proxy Handlers ──────────────────────────────────────────────────

async fn proxy_get(
    State(state): State<AppState>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
) -> Result<Response, GatewayError> {
    let url = format!("{}{}", state.config.python_core_url, uri.path());
    let resp = state.http.get(&url).send().await?;
    proxy_response(resp).await
}

async fn proxy_post(
    State(state): State<AppState>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
    body: axum::body::Body,
) -> Result<Response, GatewayError> {
    let url = format!("{}{}", state.config.python_core_url, uri.path());
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await
        .map_err(|e| GatewayError {
            status: StatusCode::BAD_REQUEST,
            message: format!("Failed to read body: {}", e),
        })?;
    let resp = state.http.post(&url)
        .body(body_bytes)
        .header("content-type", "application/json")
        .send()
        .await?;
    proxy_response(resp).await
}

async fn proxy_sse(
    State(state): State<AppState>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let url = format!("{}{}", state.config.python_core_url, uri.path());
    let http = state.http.clone();

    let stream = async_stream::stream! {
        let resp = http.get(&url).send().await;
        let mut resp = match resp {
            Ok(r) => r,
            Err(e) => {
                let evt = axum::response::sse::Event::default()
                    .data(format!("{{\"error\":\"{}\"}}", e));
                yield Ok(evt);
                return;
            }
        };

        while let Some(chunk) = resp.chunk().await.unwrap_or(None) {
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    let evt = axum::response::sse::Event::default()
                        .data(data.to_string());
                    yield Ok(evt);
                }
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

async fn catch_all_proxy(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response, GatewayError> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let url = format!("{}{}", state.config.python_core_url, path);
    let (parts, body) = req.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await
        .map_err(|e| GatewayError {
            status: StatusCode::BAD_REQUEST,
            message: format!("Failed to read body: {}", e),
        })?;

    let mut builder = match method {
        axum::http::Method::GET => state.http.get(&url),
        axum::http::Method::POST => state.http.post(&url),
        axum::http::Method::PUT => state.http.put(&url),
        axum::http::Method::DELETE => state.http.delete(&url),
        axum::http::Method::PATCH => state.http.patch(&url),
        _ => return Err(GatewayError {
            status: StatusCode::METHOD_NOT_ALLOWED,
            message: format!("Method {} not supported", method),
        }),
    };

    // Forward query string
    if let Some(qs) = parts.uri.query() {
        builder = builder.query(qs);
    }

    builder = builder.body(body_bytes);
    let resp = builder.send().await?;
    proxy_response(resp).await
}

async fn proxy_response(resp: reqwest::Response) -> Result<Response, GatewayError> {
    let status = StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut headers = HeaderMap::new();
    for (key, value) in resp.headers() {
        if let Ok(name) = key.as_str().parse::<axum::http::HeaderName>() {
            headers.insert(name, value.clone());
        }
    }

    let body = resp.bytes().await?;
    Ok((status, headers, body).into_response())
}
