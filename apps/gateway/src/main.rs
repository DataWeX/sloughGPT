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

mod compression;
mod filters;

use filters::PathPolicy;

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
    policy: PathPolicy,
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
                .unwrap_or_else(|_| default_static_dir().into()),
            listen_addr: SocketAddr::from(([0, 0, 0, 0], port)),
            policy: PathPolicy::from_env(),
        }
    }
}

/// Default document root when `MAN_STATIC_DIR` is unset: the Vite build is
/// canonical (`npm run build:vite` → `apps/web/dist-vite`); a Next asset dir
/// is only kept for checkouts that never built with Vite. An explicit
/// `MAN_STATIC_DIR` always wins.
fn default_static_dir() -> &'static str {
    if std::path::Path::new("apps/web/dist-vite").is_dir() {
        "apps/web/dist-vite"
    } else {
        "apps/web/.next/static"
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

    let app = build_router(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .expect("Failed to bind");

    info!("🌐 Gateway listening on {}", config.listen_addr);
    info!("   → Python sidecar at {}", config.python_core_url);
    if config.policy.chat_only {
        info!("   → chat-only edge mode ON (MAN_GATEWAY_CHAT_ONLY)");
    }
    if !config.policy.deny_prefixes.is_empty() {
        info!("   → deny prefixes: {:?}", config.policy.deny_prefixes);
    }

    axum::serve(listener, app).await.expect("Server failed");
}

/// Assemble the edge: gateway-owned health routes first, then the site's
/// document root when a build exists, and a byte-relay to the sidecar for
/// everything else. Extracted from `main` so tests can drive the same router.
fn build_router(state: AppState) -> Router {
    let mut app = Router::new()
        // Health & info (gateway-owned — the only endpoints understood here)
        .route("/health", get(health_check))
        .route("/health/detailed", get(detailed_health));

    let static_root = state.config.static_dir.clone();
    let static_path = std::path::Path::new(&static_root);
    if static_path.join("index.html").is_file() {
        // Document root: the whole site is static files, so the shell paints
        // without any Node process and with no dependency on the sidecar.
        // A miss falls through to `serve_document`, which routes by request
        // *kind* (navigation vs data) — see `wants_document`.
        info!(
            "serving site document root from {static_root} (SPA + API on one origin)"
        );
        let spa = Router::new()
            .fallback(serve_document)
            .with_state(state.clone());
        app = app.fallback_service(ServeDir::new(&static_root).fallback(spa));
    } else if static_path.is_dir() {
        // Legacy assets-only mode (a Next build's /_next/static): keep the
        // old nesting, relay everything else — no document root to serve.
        info!("serving static assets from {static_root} at /static");
        app = app
            .nest_service("/static", ServeDir::new(&static_root))
            .fallback(proxy_http);
    } else {
        info!(
            "static dir {static_root} absent — no site files (frontend serves itself)"
        );
        app = app.fallback(proxy_http);
    }
    app.with_state(state)
}

/// The one place a human *navigates* to API surface in a browser: the docs.
/// Path contract, not a heuristic — everything else routes by request kind.
const API_DOC_PATHS: [&str; 3] = ["/docs", "/redoc", "/openapi.json"];

fn is_api_doc_path(path: &str) -> bool {
    API_DOC_PATHS.iter().any(|p| path == *p
        || path.strip_prefix(*p).map_or(false, |rest| rest.starts_with('/')))
}

/// True when the client is asking for the SPA shell rather than data.
///
/// The SPA and the Python API own the same top-level names (`/training`,
/// `/models`, `/chat` — 600+ API paths at the root), so *path* can never
/// decide: a prefix table would shadow one side for every other client.
/// Request kind can: navigations send `Accept: text/html`, while the app's
/// own traffic (fetch/XHR/SSE) sends `*/*`, `application/json` or
/// `text/event-stream` and must always reach Python.
fn wants_document(headers: &HeaderMap, path: &str) -> bool {
    if is_api_doc_path(path) {
        return false;
    }
    headers
        .get(axum::http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map_or(false, |accept| accept.contains("text/html"))
}

/// Serve the SPA shell, else relay to the sidecar.
async fn serve_document(State(state): State<AppState>, req: Request) -> Response {
    if wants_document(req.headers(), req.uri().path()) {
        if let Some(resp) = serve_index(&state).await {
            return resp;
        }
        // index.html vanished at runtime — fall through rather than 404 a
        // navigation that the sidecar might still answer.
    }
    match proxy_http(State(state), req).await {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

/// `index.html` with no-cache: the document is the build's entry point and
/// must never outlive the assets it names (hashed filenames do the caching).
async fn serve_index(state: &AppState) -> Option<Response> {
    let path = std::path::Path::new(&state.config.static_dir).join("index.html");
    let bytes = tokio::fs::read(path).await.ok()?;
    Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .body(Body::from(bytes))
        .ok()
        .map(IntoResponse::into_response)
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
            "chat_only": state.config.policy.chat_only,
            "deny_prefixes": state.config.policy.deny_prefixes,
        },
        "sidecar": sidecar,
    }))
}

/// Unwrap the API health envelope. FastAPI answers either a bare object or
/// `{ "status": "success", "data": { ... } }` — read fields from whichever
/// level actually holds them so model_name/model_loaded aren't stuck empty.
fn parse_sidecar_health(health: &serde_json::Value) -> (bool, String) {
    let root = health.get("data").unwrap_or(health);
    let model_loaded = root
        .get("model_loaded")
        .or_else(|| health.get("model_loaded"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model_name = root
        .get("model_type")
        .or_else(|| root.get("model_name"))
        .or_else(|| root.get("model"))
        .or_else(|| health.get("model_type"))
        .or_else(|| health.get("model_name"))
        .or_else(|| health.get("model"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
        .to_string();
    (model_loaded, model_name)
}

async fn check_sidecar_health(state: &AppState) {
    let url = format!("{}/health", state.config.python_core_url);
    let (healthy, model_loaded, model_name) = match state.http.get(&url).send().await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(health) => {
                let (loaded, name) = parse_sidecar_health(&health);
                (true, loaded, name)
            }
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

async fn proxy_http(State(state): State<AppState>, req: Request) -> Result<Response, GatewayError> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // Edge policy first: rejected paths never wake Python.
    if let Err(reason) = state.config.policy.check(&path) {
        return Err(GatewayError {
            status: StatusCode::FORBIDDEN,
            message: reason.into(),
        });
    }

    let query = req
        .uri()
        .query()
        .map(|q| format!("?{q}"))
        .unwrap_or_default();
    let url = format!("{}{}{}", state.config.python_core_url, path, query);

    let (parts, body) = req.into_parts();
    let content_type = parts.headers.get("content-type").cloned();

    // Inbound cap enforced before Python wakes. 413 on overflow.
    let body_bytes = to_bytes(body, MAX_BODY_BYTES)
        .await
        .map_err(|_| GatewayError {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: format!("Request body too large (limit {} bytes)", MAX_BODY_BYTES),
        })?;

    let mut builder = match method {
        axum::http::Method::GET => state.http.get(&url),
        axum::http::Method::HEAD => state.http.head(&url),
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
    // Resume / conditional GET: forward byte-range + validator headers so
    // sidecar 206/304 semantics reach the client unchanged.
    for name in [
        "range",
        "if-range",
        "if-none-match",
        "if-modified-since",
        "if-match",
    ] {
        if let Some(v) = parts.headers.get(name) {
            builder = builder.header(name, v);
        }
    }
    // Edge owns compression: sidecar always answers identity (no double-encode).
    builder = builder.header("accept-encoding", "identity");
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
    relay_response(resp, &parts.headers, &method).await
}

async fn relay_response(
    resp: reqwest::Response,
    req_headers: &HeaderMap,
    req_method: &axum::http::Method,
) -> Result<Response, GatewayError> {
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

    // Stateless edge compression: negotiate from the *client* Accept-Encoding,
    // stream-encode, identity passthrough on skip rules. No disk, no full-body
    // buffer — reduces egress without the sidecar knowing.
    //
    // HEAD must stay identity: the body is empty while Content-Length still
    // advertises the identity size — compressing would either emit a frame
    // header alone or break zstd's pledged-size check.
    let codec = if req_method == axum::http::Method::HEAD {
        None
    } else {
        compression::negotiate(req_headers.get("accept-encoding"))
    };
    let out = compression::maybe_compress_response(out, codec, req_headers);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use tower::ServiceExt;

    #[test]
    fn parses_bare_health_object() {
        let v = serde_json::json!({
            "status": "healthy",
            "model_loaded": true,
            "model": "Qwen/Qwen2.5-0.5B-Instruct"
        });
        let (loaded, name) = parse_sidecar_health(&v);
        assert!(loaded);
        assert_eq!(name, "Qwen/Qwen2.5-0.5B-Instruct");
    }

    #[test]
    fn parses_wrapped_api_envelope() {
        let v = serde_json::json!({
            "status": "success",
            "data": {
                "status": "healthy",
                "model_loaded": true,
                "model_type": "Qwen/Qwen2.5-0.5B-Instruct",
                "device": "cpu"
            }
        });
        let (loaded, name) = parse_sidecar_health(&v);
        assert!(loaded, "model_loaded must come from data envelope");
        assert_eq!(name, "Qwen/Qwen2.5-0.5B-Instruct");
    }

    #[test]
    fn empty_name_falls_back_to_unknown() {
        let v = serde_json::json!({ "data": { "model_loaded": false, "model_type": "" } });
        let (loaded, name) = parse_sidecar_health(&v);
        assert!(!loaded);
        assert_eq!(name, "unknown");
    }

    // ── Static document root (SPA + API on one origin) ──────────────────────

    fn temp_site(files: &[(&str, &str)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("slo-gw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        for (name, content) in files {
            std::fs::write(dir.join(name), content).expect("temp file");
        }
        dir
    }

    fn test_state(dir: &std::path::Path) -> AppState {
        AppState {
            config: GatewayConfig {
                // Nothing listens on :9, so a relay attempt fails fast (502)
                // — exactly how these tests tell "proxied" from "served".
                python_core_url: "http://127.0.0.1:9".into(),
                static_dir: dir.to_string_lossy().into(),
                listen_addr: "127.0.0.1:0".parse().unwrap(),
                policy: PathPolicy::from_env(),
            },
            http: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .expect("client"),
            core_status: Arc::new(RwLock::new(CoreStatus::default())),
        }
    }

    fn get(path: &str, accept: &str) -> Request {
        Request::builder()
            .uri(path)
            .header(axum::http::header::ACCEPT, accept)
            .body(Body::empty())
            .expect("request")
    }

    async fn body_text(resp: Response) -> String {
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.expect("body");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn navigation_gets_the_spa_shell_without_the_sidecar() {
        let dir = temp_site(&[("index.html", "<html>SPA-SHELL</html>")]);
        let app = build_router(test_state(&dir));
        let resp = app.oneshot(get("/training", "text/html,application/xhtml+xml")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(axum::http::header::CONTENT_TYPE).map(|v| v.to_str().unwrap()),
            Some("text/html; charset=utf-8")
        );
        assert!(body_text(resp).await.contains("SPA-SHELL"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn data_requests_relay_to_the_sidecar_never_the_shell() {
        let dir = temp_site(&[("index.html", "<html>SPA-SHELL</html>")]);
        let app = build_router(test_state(&dir));
        let resp = app.oneshot(get("/models", "application/json")).await.unwrap();
        // Sidecar absent → the relay must be attempted (502), not shadowed by HTML.
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert!(!body_text(resp).await.contains("SPA-SHELL"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn sse_requests_relay_to_the_sidecar() {
        let dir = temp_site(&[("index.html", "<html>SPA-SHELL</html>")]);
        let app = build_router(test_state(&dir));
        let resp = app.oneshot(get("/health/stream", "text/event-stream")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn static_assets_served_from_document_root() {
        let dir = temp_site(&[
            ("index.html", "<html>SPA-SHELL</html>"),
            ("app.css", "body{}"),
        ]);
        let app = build_router(test_state(&dir));
        let resp = app.oneshot(get("/app.css", "*/*")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        assert!(ct.contains("text/css"), "got content-type {ct}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn health_stays_gateway_owned_even_with_a_document_root() {
        let dir = temp_site(&[("index.html", "<html>SPA-SHELL</html>")]);
        let app = build_router(test_state(&dir));
        let resp = app.oneshot(get("/health", "text/html")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(body_text(resp).await.contains("\"gateway\""));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn browser_docs_stay_on_the_sidecar() {
        let dir = temp_site(&[("index.html", "<html>SPA-SHELL</html>")]);
        let app = build_router(test_state(&dir));
        let resp = app.oneshot(get("/docs", "text/html")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn assets_only_dir_keeps_legacy_relay_behaviour() {
        // No index.html → not a document root: navigations relay like before.
        let dir = temp_site(&[("chunk.js", "1")]);
        let app = build_router(test_state(&dir));
        let resp = app.oneshot(get("/training", "text/html")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn request_kind_decides_document_vs_api() {
        let mut headers = HeaderMap::new();
        headers.insert(axum::http::header::ACCEPT, "text/html,application/xhtml+xml,image/webp".parse().unwrap());
        assert!(wants_document(&headers, "/training"));

        headers.insert(axum::http::header::ACCEPT, "*/*".parse().unwrap());
        assert!(!wants_document(&headers, "/training"), "fetch must reach Python");

        headers.insert(axum::http::header::ACCEPT, "application/json".parse().unwrap());
        assert!(!wants_document(&headers, "/models"));

        headers.insert(axum::http::header::ACCEPT, "text/event-stream".parse().unwrap());
        assert!(!wants_document(&headers, "/health/stream"));

        let empty = HeaderMap::new();
        assert!(!wants_document(&empty, "/training"), "no Accept → API");

        let mut html = HeaderMap::new();
        html.insert(axum::http::header::ACCEPT, "text/html".parse().unwrap());
        for path in ["/docs", "/docs/", "/redoc", "/openapi.json"] {
            assert!(!wants_document(&html, path), "{path} is API surface");
        }
    }

}
