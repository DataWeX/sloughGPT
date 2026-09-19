//! slough-gateway — transport-only edge for the Python core.
//!
//! CCGT Gateway domain: expose + optimize, never understand. This process
//! terminates TLS (rustls), caps request bodies, enforces edge timeouts,
//! relays response bytes untouched (streaming-safe: no buffering, no JSON
//! parsing, no SSE re-chunking), and serves static files. Auth, rate limits,
//! CORS semantics, and error envelopes are owned by Python — see
//! docs/PRODUCT_ENGINEERING.md ("Client / core separation").

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use futures::StreamExt;
use reqwest::Client;
use serde::Serialize;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::info;

// ── Edge policy (transport only — no model knowledge) ──────────────────────

/// Inbound body cap. Oversize → 413 before Python ever wakes.
const MAX_BODY_BYTES: usize = 1024 * 1024;
/// Total edge timeout for buffered (non-streaming) requests → 504.
const BUFFERED_TIMEOUT: Duration = Duration::from_secs(30);
/// Streaming relays (SSE) have no total timeout here — Python owns generation
/// timeouts. The reqwest client timeout below remains as a backstop.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(120);

/// Hop-by-hop headers, never forwarded upstream→client.
const HOP_HEADERS: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

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

// ── Health models (gateway-owned) ───────────────────────────────────────────

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    gateway: String,
    sidecar: CoreStatus,
    uptime_seconds: u64,
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
        if e.is_timeout() {
            return GatewayError {
                status: StatusCode::GATEWAY_TIMEOUT,
                message: "Sidecar timed out".into(),
            };
        }
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
        .timeout(CLIENT_TIMEOUT)
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
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

    let mut app = Router::new()
        // Health & info (gateway-owned — the only endpoints understood here)
        .route("/health", get(health_check))
        .route("/health/detailed", get(detailed_health))
        // Everything else: generic byte-relay to the sidecar.
        // No per-endpoint handlers — the edge never parses bodies.
        .fallback(proxy_http);

    if std::path::Path::new(&config.static_dir).is_dir() {
        info!("serving static files from {}", config.static_dir);
        app = app.nest_service("/static", ServeDir::new(&config.static_dir));
    } else {
        info!(
            "static dir {} absent — skipping /static (frontend serves itself)",
            config.static_dir
        );
    }

    let app = app
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
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

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let sidecar = state.core_status.read().await.clone();
    let uptime = START_TIME.get().map(|t| t.elapsed().as_secs()).unwrap_or(0);

    Json(HealthResponse {
        status: if sidecar.healthy {
            "ok".into()
        } else {
            "degraded".into()
        },
        gateway: "rust".into(),
        sidecar,
        uptime_seconds: uptime,
    })
}

async fn detailed_health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let sidecar = state.core_status.read().await.clone();
    let uptime = START_TIME.get().map(|t| t.elapsed().as_secs()).unwrap_or(0);

    Json(serde_json::json!({
        "gateway": {
            "status": "ok",
            "version": env!("CARGO_PKG_VERSION"),
            "uptime_seconds": uptime,
            "mode": "transport-relay",
            "max_body_bytes": MAX_BODY_BYTES,
        },
        "sidecar": sidecar,
    }))
}

async fn check_sidecar_health(state: &AppState) {
    let url = format!("{}/health", state.config.python_core_url);
    let (healthy, model_loaded, model_name) = match state.http.get(&url).send().await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(health) => (
                true,
                health
                    .get("model_loaded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                health
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            ),
            Err(_) => (false, false, String::new()),
        },
        Err(_) => (false, false, String::new()),
    };

    // Read first: skip the write lock entirely when nothing changed.
    {
        let status = state.core_status.read().await;
        if status.healthy == healthy
            && status.model_loaded == model_loaded
            && status.model_name == model_name
        {
            return;
        }
    }
    let mut status = state.core_status.write().await;
    status.healthy = healthy;
    status.last_check = Some(chrono::Utc::now());
    status.model_loaded = model_loaded;
    status.model_name = model_name;
}

// ── Generic byte-relay proxy ────────────────────────────────────────────────
//
// Forwards method + path + query + one content-type header + a size-capped
// body, then streams the sidecar's response bytes untouched. Works for JSON,
// SSE, and anything else without understanding any of it.

async fn proxy_http(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response, GatewayError> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req
        .uri()
        .query()
        .map(|q| format!("?{q}"))
        .unwrap_or_default();
    let url = format!("{}{}{}", state.config.python_core_url, path, query);

    let (parts, body) = req.into_parts();
    let content_type = parts.headers.get("content-type").cloned();

    // Inbound cap enforced before Python wakes. 413 on overflow.
    let body_bytes = to_bytes(body, MAX_BODY_BYTES).await.map_err(|_| GatewayError {
        status: StatusCode::PAYLOAD_TOO_LARGE,
        message: format!("Request body too large (limit {} bytes)", MAX_BODY_BYTES),
    })?;

    let mut builder = match method {
        axum::http::Method::GET => state.http.get(&url),
        axum::http::Method::POST => state.http.post(&url),
        axum::http::Method::PUT => state.http.put(&url),
        axum::http::Method::DELETE => state.http.delete(&url),
        axum::http::Method::PATCH => state.http.patch(&url),
        _ => {
            return Err(GatewayError {
                status: StatusCode::METHOD_NOT_ALLOWED,
                message: format!("Method {} not supported", method),
            })
        }
    };
    if let Some(ct) = content_type {
        builder = builder.header("content-type", ct);
    }
    let request = builder.body(body_bytes).build()?;

    // Streaming paths (SSE) get no total edge timeout — Python owns
    // generation timeouts. Buffered paths get a bounded edge timeout → 504.
    let streaming = path.ends_with("/stream") || path.ends_with("/regenerate");
    let resp = if streaming {
        state.http.execute(request).await?
    } else {
        tokio::time::timeout(BUFFERED_TIMEOUT, state.http.execute(request))
            .await
            .map_err(|_| GatewayError {
                status: StatusCode::GATEWAY_TIMEOUT,
                message: "Edge timeout waiting for sidecar".into(),
            })??
    };
    relay_response(resp).await
}

async fn relay_response(resp: reqwest::Response) -> Result<Response, GatewayError> {
    let status =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut headers = HeaderMap::new();
    for (key, value) in resp.headers() {
        if HOP_HEADERS.contains(&key.as_str()) {
            continue;
        }
        if let Ok(name) = key.as_str().parse::<axum::http::HeaderName>() {
            headers.append(name, value.clone());
        }
    }

    // Byte stream straight through: no buffering, no UTF-8 assumption,
    // safe for JSON, SSE, and binary alike.
    let stream = resp
        .bytes_stream()
        .map(|chunk| chunk.map_err(axum::Error::new));
    let mut out = Body::from_stream(stream).into_response();
    *out.status_mut() = status;
    out.headers_mut().extend(headers);
    Ok(out)
}
