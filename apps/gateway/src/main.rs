//! slough-gateway — transport-only edge for the Python core.
//!
//! CCGT Gateway domain: expose + optimize, never understand. This process
//! terminates TLS (rustls), caps request bodies, enforces edge timeouts,
//! relays response bytes untouched (streaming-safe: no buffering, no JSON
//! parsing, no SSE re-chunking), and serves static files. Auth and semantic
//! error envelopes are owned by Python — see docs/PRODUCT_ENGINEERING.md
//! ("Client / core separation"). Rate limiting is enforced HERE, first: the
//! full window semantics of Python's RateLimitMiddleware (route table, global
//! and workspace limits, local ×10) are ported in `ratelimit`, so floods die
//! before Python wakes; the Python copy behind us degrades to a no-op.

use axum::{
    body::{to_bytes, Body},
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use futures::StreamExt;
use ratelimit::{RateConfig, RateLimiter, RATE_LIMIT_HEADER_LIMIT, RATE_LIMIT_HEADER_REMAINING};
use reqwest::Client;
use serde::Serialize;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::{RwLock, Semaphore};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::info;

mod compression;
mod filters;
mod ratelimit;
mod supervisor;

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
    rate: RateConfig,
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
            policy: PathPolicy::from_env(),
            rate: RateConfig::from_env(),
        }
    }
}

// ── Shared State ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    config: GatewayConfig,
    http: Client,
    core_status: Arc<RwLock<CoreStatus>>,
    limiter: Arc<RateLimiter>,
    streams: Arc<Semaphore>,
}

impl AppState {
    fn new(config: GatewayConfig, http: Client) -> Self {
        // Disabled → effectively-unlimited permits (Semaphore::new rejects
        // usize::MAX); the middleware also skips acquisition when max_streams
        // is 0, so this value is never actually drawn down.
        let stream_cap = if config.rate.max_streams == 0 {
            1 << 20
        } else {
            config.rate.max_streams
        };
        Self {
            limiter: Arc::new(RateLimiter::new(config.rate)),
            streams: Arc::new(Semaphore::new(stream_cap)),
            config,
            http,
            core_status: Arc::new(RwLock::new(CoreStatus::default())),
        }
    }
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

    // Self-supervision: parent loop respawns the worker on crash so the
    // edge survives its own bugs without any init system (see supervisor.rs).
    if let supervisor::Mode::Supervisor = supervisor::mode_from_env() {
        supervisor::run().await;
        return;
    }

    let config = GatewayConfig::default();
    let _ = START_TIME.set(std::time::Instant::now());

    let http = Client::builder()
        .timeout(CLIENT_TIMEOUT)
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("Failed to create HTTP client");

    let state = AppState::new(config.clone(), http);

    // Spawn sidecar health checker
    let health_state = state.clone();
    tokio::spawn(async move {
        loop {
            check_sidecar_health(&health_state).await;
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    let app = build_router(state);

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
    if config.rate.enabled() {
        info!(
            "   → rate limit: {}/{}s per client (10× local), max {} streams (MAN_GATEWAY_RATE_LIMIT/_RATE_WINDOW, MAN_GATEWAY_MAX_STREAMS; 0 disables)",
            config.rate.global_limit,
            config.rate.window_secs,
            config.rate.max_streams,
        );
    } else {
        info!("   → rate limiting disabled (MAN_GATEWAY_RATE_LIMIT=0)");
    }

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(supervisor::shutdown_signal())
    .await
    .expect("Server failed");
}

// ── Router ──────────────────────────────────────────────────────────────────
// Shared by main() and the tests: routes + static + CORS/trace layers.

fn build_router(state: AppState) -> Router {
    let config = state.config.clone();
    let mut app = Router::new()
        // Health endpoints are contract passthroughs (see handlers below).
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

    // Admission control innermost: CORS (added next) short-circuits preflights
    // outside it, while every proxied request still passes through it.
    app = app.layer(axum::middleware::from_fn_with_state(
        state.clone(),
        edge_admission,
    ));

    app.layer(
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any),
    )
    .layer(TraceLayer::new_for_http())
    .with_state(state)
}

// ── Edge rate limiting (port of Python's RateLimitMiddleware) ───────────────

/// Streaming paths hold a concurrency permit until their body finishes.
fn is_streaming_path(path: &str) -> bool {
    path.ends_with("/stream") || path.ends_with("/regenerate")
}

/// 429 with the Python middleware's `{"detail"}` shape + rate headers.
fn reject_rate(detail: String, limit: u32, retry_after: u64) -> Response {
    let body = serde_json::json!({ "detail": detail });
    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header(RATE_LIMIT_HEADER_REMAINING, "0")
        .header(RATE_LIMIT_HEADER_LIMIT, limit.to_string())
        .header(axum::http::header::RETRY_AFTER, retry_after.to_string())
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("static 429 response")
}

/// 429 for the gateway-native concurrency cap (no window applies).
fn reject_streams() -> Response {
    let body = serde_json::json!({ "detail": "Too many concurrent streams." });
    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header(axum::http::header::RETRY_AFTER, "1")
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("static 429 response")
}

async fn edge_admission(
    State(state): State<AppState>,
    req: Request,
    next: axum::middleware::Next,
) -> Response {
    let path = req.uri().path().to_string();
    // Health probes never consume budget (supervision must stay truthful);
    // CORS preflights are short-circuited by the outer CorsLayer; static
    // assets are edge-served, never Python traffic.
    if path.starts_with("/health")
        || path.starts_with("/static/")
        || req.method() == axum::http::Method::OPTIONS
    {
        return next.run(req).await;
    }

    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0);
    let ip = peer
        .map(|p| p.ip().to_string())
        .unwrap_or_else(|| "unknown".into());
    let is_local = peer.is_some_and(|p| p.ip().is_loopback());
    let cfg = state.config.rate;
    let now = std::time::Instant::now();
    let mut rate_headers: Option<(u32, u32)> = None;

    if cfg.enabled() {
        // 1. Route-specific limit first (stricter) — Python order preserved.
        if let Some((prefix, base_limit, route_window)) = ratelimit::match_route(&path) {
            let limit = base_limit.saturating_mul(if is_local { 10 } else { 1 });
            let key = format!("r:{ip}:{prefix}");
            let check = state
                .limiter
                .check_at(&key, limit, Duration::from_secs(route_window), now);
            if check.is_err() {
                return reject_rate(
                    format!("Rate limit exceeded for {prefix}. Try again later."),
                    limit,
                    route_window,
                );
            }
        }

        // 2. Workspace limit from the Bearer JWT (best effort, never auth).
        let auth = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok());
        if let Some(ws) = ratelimit::workspace_id(auth) {
            let key = format!("w:{ws}");
            let check = state.limiter.check_at(
                &key,
                ratelimit::WORKSPACE_LIMIT,
                Duration::from_secs(cfg.window_secs),
                now,
            );
            if check.is_err() {
                return reject_rate(
                    "Workspace rate limit exceeded. Try again later.".into(),
                    ratelimit::WORKSPACE_LIMIT,
                    cfg.window_secs,
                );
            }
        }

        // 3. Global per-client limit (localhost ×10, matching Python).
        let global_limit = cfg
            .global_limit
            .saturating_mul(if is_local { 10 } else { 1 });
        let key = format!("g:{ip}");
        match state.limiter.check_at(
            &key,
            global_limit,
            Duration::from_secs(cfg.window_secs),
            now,
        ) {
            Ok(remaining) => rate_headers = Some((remaining, global_limit)),
            Err(()) => {
                return reject_rate(
                    "Too many requests. Try again later.".into(),
                    global_limit,
                    cfg.window_secs,
                )
            }
        }
    }

    // 4. Concurrency cap on streaming routes — gateway-native. The permit is
    // held until the body finishes: an SSE stream occupies inference capacity
    // for its whole life, not just until headers go out.
    let permit = if is_streaming_path(&path) && cfg.max_streams > 0 {
        match state.streams.clone().try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => return reject_streams(),
        }
    } else {
        None
    };

    let mut resp = next.run(req).await;
    if let Some(permit) = permit {
        let (parts, body) = resp.into_parts();
        let held = body.into_data_stream().map(move |chunk| {
            let _ = &permit;
            chunk
        });
        resp = Response::from_parts(parts, Body::from_stream(held));
    }
    if let Some((remaining, limit)) = rate_headers {
        // Overlay Python's values if the response came through it — the edge
        // counted the request first, its numbers are the source of truth.
        resp.headers_mut().insert(
            RATE_LIMIT_HEADER_REMAINING,
            remaining.to_string().parse().expect("numeric header"),
        );
        resp.headers_mut().insert(
            RATE_LIMIT_HEADER_LIMIT,
            limit.to_string().parse().expect("numeric header"),
        );
    }
    resp
}

// ── Health Handlers ─────────────────────────────────────────────────────────

async fn health_check(State(state): State<AppState>) -> Response {
    // Contract passthrough: the frontend reads FastAPI's whole health object
    // (model_loaded, model_type, summary, …). Relay status + body verbatim,
    // adding edge fields only when absent — the cascade must not reshape
    // app contracts.
    let url = format!("{}/health", state.config.python_core_url);
    if let Ok(resp) = state.http.get(&url).send().await {
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .cloned();
        if let Ok(body) = resp.bytes().await {
            if let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&body) {
                if let Some(obj) = value.as_object_mut() {
                    obj.entry("gateway").or_insert("rust".into());
                }
                let mut res = Json(value).into_response();
                *res.status_mut() = status;
                return res;
            }
            let mut res = Response::new(Body::from(body));
            *res.status_mut() = status;
            if let Some(ct) = ct {
                res.headers_mut()
                    .insert(axum::http::header::CONTENT_TYPE, ct);
            }
            return res;
        }
    }
    // Core unreachable: edge-owned degraded envelope (200 — host scripts probe
    // gateway liveness, not core health).
    let sidecar = state.core_status.read().await.clone();
    let uptime = START_TIME.get().map(|t| t.elapsed().as_secs()).unwrap_or(0);
    Json(HealthResponse {
        status: "degraded".into(),
        gateway: "rust".into(),
        sidecar,
        uptime_seconds: uptime,
    })
    .into_response()
}

async fn detailed_health(State(state): State<AppState>) -> Response {
    // Byte passthrough: DetailedHealth consumers expect FastAPI's monitoring
    // blob verbatim (request_count, path_latencies, health_score, …).
    let url = format!("{}/health/detailed", state.config.python_core_url);
    if let Ok(resp) = state.http.get(&url).send().await {
        let status =
            StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .cloned();
        if let Ok(body) = resp.bytes().await {
            let mut res = Response::new(Body::from(body));
            *res.status_mut() = status;
            if let Some(ct) = ct {
                res.headers_mut()
                    .insert(axum::http::header::CONTENT_TYPE, ct);
            }
            return res;
        }
    }
    // Core unreachable: edge-owned envelope (existing shape).
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
            "rate_limit": {
                "limit": state.config.rate.global_limit,
                "window_secs": state.config.rate.window_secs,
                "max_streams": state.config.rate.max_streams,
            },
        },
        "sidecar": sidecar,
    }))
    .into_response()
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
    let streaming = is_streaming_path(&path);
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

    /// Test state with the core pointed at a closed port — handler tests
    /// never race a live FastAPI.
    fn edge_state() -> AppState {
        edge_state_rate(RateConfig::default())
    }

    /// Same, with explicit rate-limit config for admission tests.
    fn edge_state_rate(rate: RateConfig) -> AppState {
        let config = GatewayConfig {
            python_core_url: "http://127.0.0.1:9".into(),
            rate,
            ..Default::default()
        };
        AppState::new(config, Client::new())
    }

    #[tokio::test]
    async fn preflight_options_terminated_at_edge_with_cors() {
        let app = build_router(edge_state());
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/api/sessions")
            .header("origin", "http://localhost:3000")
            .header("access-control-request-method", "POST")
            .header("access-control-request-headers", "content-type")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(matches!(
            resp.status(),
            StatusCode::NO_CONTENT | StatusCode::OK
        ));
        let h = resp.headers();
        assert_eq!(
            h.get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("*")
        );
        assert!(h.get("access-control-allow-headers").is_some());
    }

    #[tokio::test]
    async fn health_falls_back_to_edge_envelope_when_core_down() {
        let app = build_router(edge_state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let h = resp.headers();
        assert_eq!(
            h.get("access-control-expose-headers")
                .and_then(|v| v.to_str().ok()),
            Some("*")
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["gateway"], "rust");
    }

    #[tokio::test]
    async fn health_passes_core_contract_through_with_additive_gateway() {
        let stub = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({
                    "status": "ok",
                    "model_loaded": true,
                    "model_type": "slonet",
                    "summary": "ready",
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, stub).await.unwrap();
        });

        let mut state = edge_state();
        state.config.python_core_url = format!("http://{addr}");
        let app = build_router(state);
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // FastAPI contract intact at top level…
        assert_eq!(v["model_loaded"], true);
        assert_eq!(v["model_type"], "slonet");
        assert_eq!(v["summary"], "ready");
        // …plus the additive edge marker.
        assert_eq!(v["gateway"], "rust");
    }

    // ── Admission control (Python port) ─────────────────────────────────────

    #[tokio::test]
    async fn global_window_rejects_over_limit_with_python_shape() {
        let app = build_router(edge_state_rate(RateConfig {
            global_limit: 2,
            window_secs: 60,
            max_streams: 0,
        }));
        // Two admitted requests reach the (down) core → 502, not 429.
        for _ in 0..2 {
            let req = Request::builder()
                .uri("/api/nope")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }
        // Third exceeds the window → Python-shaped 429.
        let req = Request::builder()
            .uri("/api/nope")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let h = resp.headers();
        assert_eq!(
            h.get(RATE_LIMIT_HEADER_REMAINING)
                .and_then(|v| v.to_str().ok()),
            Some("0")
        );
        assert_eq!(
            h.get(RATE_LIMIT_HEADER_LIMIT).and_then(|v| v.to_str().ok()),
            Some("2")
        );
        assert_eq!(
            h.get("retry-after").and_then(|v| v.to_str().ok()),
            Some("60")
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["detail"], "Too many requests. Try again later.");
    }

    #[tokio::test]
    async fn route_limits_are_stricter_than_global() {
        let app = build_router(edge_state_rate(RateConfig {
            global_limit: 1000,
            window_secs: 60,
            max_streams: 0,
        }));
        for _ in 0..10 {
            let req = Request::builder()
                .uri("/chat/stream")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }
        // 11th hits the /chat/stream table limit (10/60), long before global.
        let req = Request::builder()
            .uri("/chat/stream")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let h = resp.headers();
        assert_eq!(
            h.get(RATE_LIMIT_HEADER_LIMIT).and_then(|v| v.to_str().ok()),
            Some("10")
        );
        assert_eq!(
            h.get("retry-after").and_then(|v| v.to_str().ok()),
            Some("60")
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            v["detail"],
            "Rate limit exceeded for /chat/stream. Try again later."
        );
    }

    #[tokio::test]
    async fn health_and_options_bypass_the_limiter() {
        let app = build_router(edge_state_rate(RateConfig {
            global_limit: 1,
            window_secs: 60,
            max_streams: 0,
        }));
        // Burn the single token.
        let req = Request::builder()
            .uri("/api/x")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        // Health still answers — supervision must stay truthful under load.
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // CORS preflight never sees the limiter (CorsLayer is outer).
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/api/sessions")
            .header("origin", "http://localhost:3000")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(matches!(
            resp.status(),
            StatusCode::NO_CONTENT | StatusCode::OK
        ));
    }

    #[tokio::test]
    async fn stream_concurrency_capped_until_body_finishes() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let hit = Arc::new(AtomicBool::new(false));
        let stub = Router::new().route(
            "/chat/stream",
            get({
                let hit = hit.clone();
                move || {
                    let hit = hit.clone();
                    async move {
                        hit.store(true, Ordering::SeqCst);
                        // Never completes: the stream holds its permit.
                        std::future::pending::<()>().await;
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, stub).await.unwrap();
        });

        let mut state = edge_state_rate(RateConfig {
            global_limit: 100_000,
            window_secs: 60,
            max_streams: 1,
        });
        state.config.python_core_url = format!("http://{addr}");
        let app = build_router(state);

        let first = app.clone();
        tokio::spawn(async move {
            let req = Request::builder()
                .uri("/chat/stream")
                .body(Body::empty())
                .unwrap();
            let _ = first.oneshot(req).await;
        });
        for _ in 0..10_000 {
            if hit.load(Ordering::SeqCst) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            hit.load(Ordering::SeqCst),
            "first stream never reached stub"
        );

        // Second stream while the first still holds the only permit → 429.
        let req = Request::builder()
            .uri("/chat/stream")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["detail"], "Too many concurrent streams.");
    }

    #[tokio::test]
    async fn zero_limit_disables_window_checks_entirely() {
        let app = build_router(edge_state_rate(RateConfig {
            global_limit: 0,
            window_secs: 60,
            max_streams: 0,
        }));
        // 350 > default global (300) and > route table (10): if any window
        // check were still live, this would429 long before the end.
        for _ in 0..350 {
            let req = Request::builder()
                .uri("/chat/stream")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }
    }
}
