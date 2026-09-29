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
use std::{
    fs::{File, OpenOptions},
    io::Write,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::info;

mod breaker;
mod compression;
mod filters;
mod ratelimit;
mod stats;
mod streamctl;
mod supervisor;

use breaker::{BreakerConfig, CircuitBreaker};
use filters::PathPolicy;
use stats::{ClientSeen, EdgeStats};
use streamctl::StreamControl;

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
    breaker: BreakerConfig,
    /// JSONL access log path; empty = off.
    access_log: String,
}

/// Unset → `logs/gateway-access.jsonl`; `0` or empty → off.
fn access_log_from_env() -> String {
    match std::env::var("MAN_GATEWAY_ACCESS_LOG") {
        Ok(v) if v.is_empty() || v == "0" => String::new(),
        Ok(v) => v,
        Err(_) => "logs/gateway-access.jsonl".into(),
    }
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
            breaker: BreakerConfig::from_env(),
            access_log: access_log_from_env(),
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
    /// Adaptive stream admission: AIMD ceiling ≤ `MAN_GATEWAY_MAX_STREAMS`.
    streams: Arc<StreamControl>,
    stats: Arc<EdgeStats>,
    breaker: Arc<CircuitBreaker>,
    /// Append-only JSONL access log; `None` when disabled or unopenable.
    /// Arc'd so `AppState` stays `Clone` while all requests share one handle.
    access: Option<Arc<Mutex<File>>>,
}

/// Create parent dirs, open append-only. Failure degrades to no logging —
/// the gateway must never die on its own log file.
fn open_access_log(path: &str) -> Option<Arc<Mutex<File>>> {
    if path.is_empty() {
        return None;
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    match OpenOptions::new().create(true).append(true).open(path) {
        Ok(f) => Some(Arc::new(Mutex::new(f))),
        Err(e) => {
            tracing::warn!("access log {path} unavailable ({e}) — logging off");
            None
        }
    }
}

impl AppState {
    fn new(config: GatewayConfig, http: Client) -> Self {
        Self {
            limiter: Arc::new(RateLimiter::new(config.rate)),
            streams: Arc::new(StreamControl::new(config.rate.max_streams)),
            stats: Arc::new(EdgeStats::new()),
            breaker: Arc::new(CircuitBreaker::new(config.breaker)),
            access: open_access_log(&config.access_log),
            config,
            http,
            core_status: Arc::new(RwLock::new(CoreStatus::default())),
        }
    }

    /// One flushed JSONL line per request — the history `/gateway/stats`
    /// (in-memory, reset on restart) structurally cannot keep.
    // Flat args: a per-line struct would add ceremony without clarity.
    #[allow(clippy::too_many_arguments)]
    fn access_log(
        &self,
        method: &axum::http::Method,
        path: &str,
        route: &str,
        status: u16,
        started: Instant,
        shed: Option<&str>,
        client: &str,
    ) {
        let Some(file) = &self.access else { return };
        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let line = serde_json::json!({
            "ts_ms": ts_ms,
            "ms": (ms * 10.0).round() / 10.0,
            "method": method.as_str(),
            "path": path,
            "route": route,
            "status": status,
            "shed": shed,
            "client": client,
        });
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
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
    if config.breaker.enabled() {
        info!(
            "   → breaker: {} consecutive upstream failures → open {}s (MAN_GATEWAY_BREAKER_FAILURES/_OPEN; 0 disables)",
            config.breaker.failures, config.breaker.open_secs
        );
    } else {
        info!("   → circuit breaker disabled (MAN_GATEWAY_BREAKER_FAILURES=0)");
    }
    if config.rate.max_streams > 0 {
        info!(
            "   → streams: adaptive cap ≤ {} (AIMD, 2s windows; halves at ≥12.5% stream failures; MAN_GATEWAY_MAX_STREAMS)",
            config.rate.max_streams
        );
    }
    if config.access_log.is_empty() {
        info!("   → access log disabled (MAN_GATEWAY_ACCESS_LOG=0)");
    } else {
        info!(
            "   → access log: {} (MAN_GATEWAY_ACCESS_LOG; 0 disables)",
            config.access_log
        );
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
        // Edge-owned performance stats — never proxied, never shaped by Python.
        .route("/gateway/stats", get(gateway_stats))
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

/// 503 shed while the circuit breaker is open — fail fast instead of
/// queueing behind a dead core's edge timeout.
fn reject_breaker(retry_after: u64) -> Response {
    let body = serde_json::json!({ "detail": "Upstream unavailable." });
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(
            axum::http::header::RETRY_AFTER,
            retry_after.max(1).to_string(),
        )
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("static 503 response")
}

async fn edge_admission(
    State(state): State<AppState>,
    req: Request,
    next: axum::middleware::Next,
) -> Response {
    let started = Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let route = stats::route_key(&path);
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0);
    let ip = peer
        .map(|p| p.ip().to_string())
        .unwrap_or_else(|| "unknown".into());
    let is_local = peer.is_some_and(|p| p.ip().is_loopback());

    // Health probes never consume budget (supervision must stay truthful);
    // the edge's own stats endpoint must stay readable under load; CORS
    // preflights are short-circuited by the outer CorsLayer; static assets
    // are edge-served, never Python traffic. Still logged — the access log
    // exists to show what actually consumes the edge.
    if path.starts_with("/health")
        || path.starts_with("/gateway/")
        || path.starts_with("/static/")
        || method == axum::http::Method::OPTIONS
    {
        let resp = next.run(req).await;
        state.access_log(
            &method,
            &path,
            &route,
            resp.status().as_u16(),
            started,
            None,
            &ip,
        );
        return resp;
    }

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
                state.stats.shed_429(&route);
                state.stats.client_seen(&ip, ClientSeen::Shed429);
                state.access_log(&method, &path, &route, 429, started, Some("rate"), &ip);
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
                state.stats.shed_429(&route);
                state.stats.client_seen(&ip, ClientSeen::Shed429);
                state.access_log(&method, &path, &route, 429, started, Some("rate"), &ip);
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
                state.stats.shed_429(&route);
                state.stats.client_seen(&ip, ClientSeen::Shed429);
                state.access_log(&method, &path, &route, 429, started, Some("rate"), &ip);
                return reject_rate(
                    "Too many requests. Try again later.".into(),
                    global_limit,
                    cfg.window_secs,
                );
            }
        }
    }

    // 4. Circuit breaker: while open, shed instantly — do not spend an edge
    // timeout on a core the edge has already watched fail K times.
    if let Err(retry) = state.breaker.allow(now) {
        state.stats.shed_503(&route);
        state.stats.client_seen(&ip, ClientSeen::Shed503);
        state.access_log(&method, &path, &route, 503, started, Some("breaker"), &ip);
        return reject_breaker(retry);
    }

    // 5. Adaptive concurrency cap on streaming routes — gateway-native.
    // AIMD: tune against observed stream failures first, then CAS a slot
    // against the in-flight gauge. The guard is held until the body
    // finishes: an SSE stream occupies inference capacity for its whole
    // life, not just until headers go out.
    let guard = if is_streaming_path(&path) && cfg.max_streams > 0 {
        state.streams.tune(now);
        match stats::try_stream_slot(&state.stats, state.streams.limit()) {
            Some(guard) => {
                state.streams.record_start();
                Some(guard)
            }
            None => {
                state.stats.shed_429(&route);
                state.stats.client_seen(&ip, ClientSeen::Shed429);
                state.access_log(&method, &path, &route, 429, started, Some("streams"), &ip);
                return reject_streams();
            }
        }
    } else {
        // In-flight gauge tracks streaming requests only: dropped when the
        // body finishes, aborts, or the request errors downstream.
        None
    };

    let mut resp = next.run(req).await;
    if let Some(guard) = guard {
        let (parts, body) = resp.into_parts();
        let held = body.into_data_stream().map(move |chunk| {
            let _ = &guard;
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
    state
        .stats
        .client_seen(&ip, ClientSeen::Response(resp.status().as_u16()));
    state.access_log(
        &method,
        &path,
        &route,
        resp.status().as_u16(),
        started,
        None,
        &ip,
    );
    resp
}

// ── Health Handlers ─────────────────────────────────────────────────────────

/// Edge-owned performance stats — the view FastAPI structurally cannot have
/// (it only sees traffic that reached Python; this sees everything, including
/// what the edge shed). Additive endpoint; never proxied, never reshaped.
async fn gateway_stats(State(state): State<AppState>) -> Response {
    let uptime = START_TIME.get().map(|t| t.elapsed().as_secs()).unwrap_or(0);
    let (breaker_state, retry) = state.breaker.status(Instant::now());
    let snap = state.stats.snapshot();
    let body = serde_json::json!({
        "gateway": {
            "uptime_seconds": uptime,
            "in_flight_streams": state.stats.in_flight(),
            "stream_limit": state.streams.limit(),
            "stream_max": state.streams.max(),
            "breaker": {
                "state": breaker_state,
                "retry_after_secs": retry,
                "failures_to_trip": state.config.breaker.failures,
                "open_secs": state.config.breaker.open_secs,
            },
        },
        "totals": snap["totals"],
        "routes": snap["routes"],
        "clients": snap["clients"],
    });
    Json(body).into_response()
}

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
            "breaker": {
                "failures": state.config.breaker.failures,
                "open_secs": state.config.breaker.open_secs,
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
    let start = Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let route = stats::route_key(&path);

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
    state.stats.add_bytes_in(&route, body_bytes.len());

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
    // Every transport failure feeds the breaker — edge-observed outcomes only.
    let streaming = is_streaming_path(&path);
    let resp = if streaming {
        match state.http.execute(request).await {
            Ok(resp) => resp,
            Err(e) => {
                state.stats.add_upstream_err(&route);
                state.breaker.record_failure();
                state.streams.record_failure();
                return Err(e.into());
            }
        }
    } else {
        match tokio::time::timeout(BUFFERED_TIMEOUT, state.http.execute(request)).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                state.stats.add_upstream_err(&route);
                state.breaker.record_failure();
                return Err(e.into());
            }
            Err(_) => {
                state.stats.add_upstream_err(&route);
                state.breaker.record_failure();
                return Err(GatewayError {
                    status: StatusCode::GATEWAY_TIMEOUT,
                    message: "Edge timeout waiting for sidecar".into(),
                });
            }
        }
    };
    // Upstream answered: 5xx counts as a breaker failure, anything else
    // (incl. 4xx — client's problem, not the core's) is a success. On
    // streaming paths 5xx also feeds the adaptive window — the core is
    // alive (headers arrived) but failing streams.
    state.breaker.observe(resp.status().as_u16());
    if streaming && resp.status().as_u16() >= 500 {
        state.streams.record_failure();
    }
    let out = relay_response(resp, &parts.headers, &method, &state.stats, &route).await?;
    state
        .stats
        .record_upstream(&route, out.status().as_u16(), start.elapsed());

    // Observe wire bytes (post-compression) — compression.rs drops
    // Content-Length on the compressed path; the identity path's header
    // survives the wrap unchanged because the bytes are identical.
    let (resp_parts, body) = out.into_parts();
    let wstats = state.stats.clone();
    let wroute = route.clone();
    let counted = body.into_data_stream().map(move |chunk| {
        if let Ok(b) = &chunk {
            wstats.add_wire(&wroute, b.len());
        }
        chunk
    });
    Ok(Response::from_parts(resp_parts, Body::from_stream(counted)))
}

async fn relay_response(
    resp: reqwest::Response,
    req_headers: &HeaderMap,
    req_method: &axum::http::Method,
    stats: &Arc<EdgeStats>,
    route: &str,
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
    // safe for JSON, SSE, and binary alike. Chunk sizes feed the identity
    // byte counter (the no-compression baseline).
    let istats = stats.clone();
    let iroute = route.to_string();
    let stream = resp.bytes_stream().map(move |chunk| {
        if let Ok(b) = &chunk {
            istats.add_identity(&iroute, b.len());
        }
        chunk.map_err(axum::Error::new)
    });
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
    /// Breaker disabled by default so multi-request tests against the closed
    /// port aren't tripped mid-sequence; breaker tests opt in explicitly.
    fn edge_state_rate(rate: RateConfig) -> AppState {
        let config = GatewayConfig {
            python_core_url: "http://127.0.0.1:9".into(),
            rate,
            breaker: BreakerConfig {
                failures: 0,
                open_secs: 0,
            },
            access_log: String::new(),
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

    // ── Stats + circuit breaker (performance control plane) ────────────────

    #[tokio::test]
    async fn stats_endpoint_reports_shape_and_bypasses_limiter() {
        // limit 1: burn the only token on a proxied request…
        let app = build_router(edge_state_rate(RateConfig {
            global_limit: 1,
            window_secs: 60,
            max_streams: 0,
        }));
        let req = Request::builder()
            .uri("/api/x")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        // …then the window is exhausted, but stats must stay readable.
        let req = Request::builder()
            .uri("/gateway/stats")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(v["gateway"]["breaker"]["state"].is_string());
        assert_eq!(v["gateway"]["in_flight_streams"], 0);
        assert!(v["totals"]["requests"].is_number());
        // The failed proxied request is visible as an upstream error.
        let routes = v["routes"].as_array().unwrap();
        let api = routes.iter().find(|r| r["route"] == "/api").unwrap();
        assert_eq!(api["upstream_err"], 1);
        assert_eq!(v["totals"]["upstream_err"], 1);
    }

    #[tokio::test]
    async fn breaker_sheds_503_after_consecutive_failures() {
        let config = GatewayConfig {
            python_core_url: "http://127.0.0.1:9".into(),
            rate: RateConfig {
                global_limit: 10_000,
                window_secs: 60,
                max_streams: 0,
            },
            breaker: BreakerConfig {
                failures: 2,
                open_secs: 60,
            },
            access_log: String::new(),
            ..Default::default()
        };
        let app = build_router(AppState::new(config, Client::new()));
        // Two failures → breaker opens.
        for _ in 0..2 {
            let req = Request::builder()
                .uri("/api/x")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }
        // Third → instant shed, no connect attempt to the dead core.
        let req = Request::builder()
            .uri("/api/x")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let h = resp.headers();
        let retry: u64 = h
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .unwrap()
            .parse()
            .unwrap();
        assert!((1..=60).contains(&retry));
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["detail"], "Upstream unavailable.");
        // Stats see the shed and the open breaker.
        let req = Request::builder()
            .uri("/gateway/stats")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["gateway"]["breaker"]["state"], "open");
        assert_eq!(v["totals"]["shed_503"], 1);
        assert_eq!(v["totals"]["upstream_err"], 2);
    }

    // ── Access log ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn access_log_records_exempt_admitted_and_shed_lines() {
        let path = std::env::temp_dir().join(format!("slo-gw-access-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let config = GatewayConfig {
            python_core_url: "http://127.0.0.1:9".into(),
            rate: RateConfig {
                global_limit: 1,
                window_secs: 60,
                max_streams: 0,
            },
            breaker: BreakerConfig {
                failures: 0,
                open_secs: 0,
            },
            access_log: path.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let app = build_router(AppState::new(config, Client::new()));

        // Exempt endpoint (readable under load) — logged with its real status.
        let req = Request::builder()
            .uri("/gateway/stats")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::OK
        );
        // Admitted → dead core → 502.
        let req = Request::builder()
            .uri("/api/x")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::BAD_GATEWAY
        );
        // Second request exceeds global_limit=1 → rate shed.
        let req = Request::builder()
            .uri("/api/y")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::TOO_MANY_REQUESTS
        );

        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid JSONL"))
            .collect();
        assert_eq!(lines.len(), 3, "one line per request: {text}");

        assert_eq!(lines[0]["status"], 200);
        assert!(lines[0]["shed"].is_null());
        assert_eq!(lines[0]["path"], "/gateway/stats");
        assert_eq!(lines[0]["route"], "/gateway");
        assert!(lines[0]["ms"].is_number());
        assert!(lines[0]["ts_ms"].is_number());

        assert_eq!(lines[1]["status"], 502);
        assert!(lines[1]["shed"].is_null());
        assert_eq!(lines[1]["route"], "/api");

        assert_eq!(lines[2]["status"], 429);
        assert_eq!(lines[2]["shed"], "rate");
        assert_eq!(lines[2]["method"], "GET");

        let _ = std::fs::remove_file(&path);
    }

    // ── Per-client stats ───────────────────────────────────────────────────

    #[tokio::test]
    async fn per_client_stats_attribute_responses_and_sheds() {
        let config = GatewayConfig {
            python_core_url: "http://127.0.0.1:9".into(),
            rate: RateConfig {
                global_limit: 1,
                window_secs: 60,
                max_streams: 0,
            },
            breaker: BreakerConfig {
                failures: 0,
                open_secs: 0,
            },
            access_log: String::new(),
            ..Default::default()
        };
        let app = build_router(AppState::new(config, Client::new()));

        fn from_client(ip: [u8; 4], uri: &str) -> Request {
            let mut req = Request::builder().uri(uri).body(Body::empty()).unwrap();
            req.extensions_mut()
                .insert(ConnectInfo(SocketAddr::from((ip, 40000u16))));
            req
        }

        // Client A: admitted → dead core → 502 (counts as 5xx).
        let resp = app
            .clone()
            .oneshot(from_client([10, 0, 0, 1], "/api/x"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        // Client B: one admitted → 502, second request rate-shed (limit 1).
        let resp = app
            .clone()
            .oneshot(from_client([10, 0, 0, 2], "/api/x"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let resp = app
            .clone()
            .oneshot(from_client([10, 0, 0, 2], "/api/y"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        // Exempt stats fetch (no ConnectInfo) must not appear as a client —
        // edge-internal polls are overhead, not consumers.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/gateway/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let clients = v["clients"].as_array().unwrap();
        assert_eq!(clients.len(), 2, "A + B only, got: {clients:?}");
        assert!(clients.iter().all(|c| c["client"] != "unknown"));

        let a = clients.iter().find(|c| c["client"] == "10.0.0.1").unwrap();
        assert_eq!(a["requests"], 1);
        assert_eq!(a["5xx"], 1);
        assert_eq!(a["shed_429"], 0);

        let b = clients.iter().find(|c| c["client"] == "10.0.0.2").unwrap();
        assert_eq!(b["requests"], 1);
        assert_eq!(b["5xx"], 1);
        assert_eq!(b["shed_429"], 1);
    }

    // ── Adaptive stream limits (AIMD) ──────────────────────────────────────

    #[tokio::test]
    async fn adaptive_stream_limit_halves_after_failures() {
        let config = GatewayConfig {
            python_core_url: "http://127.0.0.1:9".into(),
            rate: RateConfig {
                global_limit: 100_000,
                window_secs: 60,
                max_streams: 8,
            },
            // Breaker off: we want the streams *admitted and failed* to feed
            // the AIMD window — the breaker would shed them after 5.
            breaker: BreakerConfig {
                failures: 0,
                open_secs: 0,
            },
            access_log: String::new(),
            ..Default::default()
        };
        let app = build_router(AppState::new(config, Client::new()));

        let stats = |app: axum::Router| async move {
            let resp = app
                .oneshot(
                    Request::builder()
                        .uri("/gateway/stats")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
            serde_json::from_slice::<serde_json::Value>(&body).unwrap()
        };

        assert_eq!(stats(app.clone()).await["gateway"]["stream_limit"], 8);

        // 8 admitted streams, all fail against the dead core — window:
        // starts=8, failures=8 (100% ≥ 12.5%).
        for _ in 0..8 {
            let req = Request::builder()
                .method("POST")
                .uri("/chat/stream")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }

        // Past the 2s tune interval → the next admission evaluates the
        // window and halves the effective cap: 8 → 4.
        tokio::time::sleep(Duration::from_millis(2200)).await;
        let req = Request::builder()
            .method("POST")
            .uri("/chat/stream")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let v = stats(app).await;
        assert_eq!(
            v["gateway"]["stream_limit"], 4,
            "AIMD halved the cap after a 100% failure window: {v}"
        );
        assert_eq!(v["gateway"]["stream_max"], 8, "ceiling unchanged");
    }
}
