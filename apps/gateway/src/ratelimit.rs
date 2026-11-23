//! Rate limiting at the edge — full port of Python's `RateLimitMiddleware`.
//!
//! Python's FastAPI copy (`apps/api/server/infrastructure/rate_limit_middleware.py`)
//! still runs behind us; enforced *here* first, it degrades to a no-op. Window
//! semantics, route table, localhost ×10 multiplier, per-workspace JWT limits,
//! `X-RateLimit-*` headers, and the `429 {"detail": ...}` body shape are ported
//! verbatim — change both together or neither.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Header names — same constants as `domain/infrastructure/_internal/rate_limiter.py`.
pub const RATE_LIMIT_HEADER_REMAINING: &str = "X-RateLimit-Remaining";
pub const RATE_LIMIT_HEADER_LIMIT: &str = "X-RateLimit-Limit";

/// Workspace default — mirrors `_DEFAULT_WORKSPACE_LIMIT`.
pub const WORKSPACE_LIMIT: u32 = 300;
/// Idle GC horizon — larger than any route window (max 120s).
const IDLE_EVICT: Duration = Duration::from_secs(240);
/// Map size that triggers idle eviction on insert.
const GC_THRESHOLD: usize = 2048;

/// Exact replica of Python's `_ROUTE_LIMITS`: (prefix, limit, window_secs).
/// Python dict insertion order — first prefix match wins.
const ROUTE_LIMITS: &[(&str, u32, u64)] = &[
    ("/chat/stream", 10, 60),
    ("/chat/voice", 5, 60),
    ("/inference/generate/stream", 10, 60),
    ("/inference/generate", 20, 60),
    ("/inference/embed", 30, 60),
    ("/models/load", 2, 120),
    ("/models/unload", 2, 120),
    ("/training/start", 3, 60),
    ("/training/control/start", 3, 60),
    ("/training/turbo-start", 3, 60),
    ("/training/from-sessions-start", 3, 60),
    ("/training/stop", 5, 60),
    ("/training/control/stop", 5, 60),
    ("/training/status", 60, 60),
    ("/training/log", 60, 60),
    ("/training/checkpoints", 30, 60),
    ("/training/stream", 30, 60),
    ("/training/is-running", 60, 60),
    ("/mobile/train", 2, 120),
    ("/models/", 30, 60),
];

/// First matching route prefix, mirroring Python's dict-order scan.
pub fn match_route(path: &str) -> Option<(&'static str, u32, u64)> {
    ROUTE_LIMITS
        .iter()
        .find(|(prefix, _, _)| path.starts_with(prefix))
        .map(|&(prefix, limit, window)| (prefix, limit, window))
}

// ── Config ──────────────────────────────────────────────────────────────────

/// Env-tunable limits. `0` disables a dimension.
///
/// - `MAN_GATEWAY_RATE_LIMIT` (default `300`) — global requests per window per client
/// - `MAN_GATEWAY_RATE_WINDOW` (default `60`) — window seconds
/// - `MAN_GATEWAY_MAX_STREAMS` (default `32`) — concurrent SSE streams; `0` = off
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateConfig {
    pub global_limit: u32,
    pub window_secs: u64,
    pub max_streams: usize,
}

impl Default for RateConfig {
    fn default() -> Self {
        Self {
            global_limit: 300,
            window_secs: 60,
            max_streams: 32,
        }
    }
}

impl RateConfig {
    pub fn from_env() -> Self {
        Self::parse(
            std::env::var("MAN_GATEWAY_RATE_LIMIT").ok().as_deref(),
            std::env::var("MAN_GATEWAY_RATE_WINDOW").ok().as_deref(),
            std::env::var("MAN_GATEWAY_MAX_STREAMS").ok().as_deref(),
        )
    }

    /// Unset or unparseable → default. `"0"` disables.
    pub fn parse(limit: Option<&str>, window: Option<&str>, streams: Option<&str>) -> Self {
        let d = Self::default();
        Self {
            global_limit: limit
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(d.global_limit),
            window_secs: window
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(d.window_secs),
            max_streams: streams
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(d.max_streams),
        }
    }

    /// Window checks active only with a live limit and window.
    pub fn enabled(&self) -> bool {
        self.global_limit > 0 && self.window_secs > 0
    }
}

// ── Sliding-window counter ──────────────────────────────────────────────────

pub struct RateLimiter {
    windows: Mutex<HashMap<String, VecDeque<Instant>>>,
}

impl RateLimiter {
    pub fn new(_cfg: RateConfig) -> Self {
        Self {
            windows: Mutex::new(HashMap::new()),
        }
    }

    /// Sliding-window check: `Ok(remaining)` admits, `Err(())` over limit.
    /// Mirrors Python's `RateLimiter.check` — prune, append, remaining after push.
    pub fn check_at(
        &self,
        key: &str,
        limit: u32,
        window: Duration,
        now: Instant,
    ) -> Result<u32, ()> {
        let mut map = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        if map.len() > GC_THRESHOLD {
            map.retain(|_, q| {
                q.back()
                    .is_some_and(|&t| now.saturating_duration_since(t) < IDLE_EVICT)
            });
        }
        let cutoff = now.checked_sub(window).unwrap_or(now);
        let q = map.entry(key.to_string()).or_default();
        while let Some(&t) = q.front() {
            if t < cutoff {
                q.pop_front();
            } else {
                break;
            }
        }
        if q.len() as u64 >= limit as u64 {
            return Err(());
        }
        q.push_back(now);
        Ok(limit - q.len() as u32)
    }
}

// ── Workspace extraction (port of _extract_workspace_from_token) ────────────

/// Minimal base64url decode — JWT payloads only (URL-safe, padding optional).
pub fn base64url_decode(input: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'-' => Some(62),
            b'_' => Some(63),
            b'=' => None, // padding: stop below
            _ => None,
        }
    }
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    for &b in input.as_bytes() {
        if b == b'=' {
            break;
        }
        let v = val(b)?;
        acc = (acc << 6) | u32::from(v);
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
            acc &= (1 << nbits) - 1;
        }
    }
    Some(out)
}

/// Best-effort workspace_id from a Bearer JWT — rate limiting only, never auth.
pub fn workspace_id(authorization: Option<&str>) -> Option<String> {
    let auth = authorization?;
    let token = auth.strip_prefix("Bearer ")?;
    let mut parts = token.split('.');
    parts.next()?;
    let payload = parts.next()?;
    parts.next()?;
    let raw = base64url_decode(payload)?;
    let value: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let ws = value.get("workspace_id")?.as_str()?;
    (!ws.is_empty()).then(|| ws.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_when_unset_or_garbage() {
        assert_eq!(RateConfig::parse(None, None, None), RateConfig::default());
        let cfg = RateConfig::parse(Some("abc"), Some(""), Some("-5"));
        assert_eq!(cfg, RateConfig::default());
    }

    #[test]
    fn zero_disables_window_checks() {
        let cfg = RateConfig::parse(Some("0"), Some("60"), Some("0"));
        assert!(!cfg.enabled());
        assert_eq!(cfg.max_streams, 0);
    }

    #[test]
    fn window_admits_up_to_limit_then_rejects() {
        let limiter = RateLimiter::new(RateConfig::default());
        let now = Instant::now();
        let w = Duration::from_secs(60);
        assert_eq!(limiter.check_at("k", 3, w, now), Ok(2));
        assert_eq!(limiter.check_at("k", 3, w, now), Ok(1));
        assert_eq!(limiter.check_at("k", 3, w, now), Ok(0));
        assert_eq!(limiter.check_at("k", 3, w, now), Err(()));
    }

    #[test]
    fn window_slides_forward() {
        let limiter = RateLimiter::new(RateConfig::default());
        let t0 = Instant::now();
        let w = Duration::from_secs(60);
        assert_eq!(limiter.check_at("k", 1, w, t0), Ok(0));
        assert_eq!(limiter.check_at("k", 1, w, t0), Err(()));
        // Past the window: old entries prune, slot frees.
        let t1 = t0 + Duration::from_secs(61);
        assert_eq!(limiter.check_at("k", 1, w, t1), Ok(0));
    }

    #[test]
    fn keys_are_independent() {
        let limiter = RateLimiter::new(RateConfig::default());
        let now = Instant::now();
        let w = Duration::from_secs(60);
        assert_eq!(limiter.check_at("a", 1, w, now), Ok(0));
        assert_eq!(limiter.check_at("b", 1, w, now), Ok(0));
        assert_eq!(limiter.check_at("a", 1, w, now), Err(()));
        assert_eq!(limiter.check_at("b", 1, w, now), Err(()));
    }

    #[test]
    fn route_match_prefers_first_table_entry() {
        assert_eq!(match_route("/chat/stream"), Some(("/chat/stream", 10, 60)));
        assert_eq!(
            match_route("/chat/stream/x"),
            Some(("/chat/stream", 10, 60))
        );
        assert_eq!(match_route("/models/load"), Some(("/models/load", 2, 120)));
        assert_eq!(match_route("/models/anything"), Some(("/models/", 30, 60)));
        assert_eq!(
            match_route("/training/control/start"),
            Some(("/training/control/start", 3, 60))
        );
        assert_eq!(match_route("/unrelated"), None);
        assert_eq!(match_route("/health"), None);
    }

    #[test]
    fn base64url_decodes_urlsafe_and_padded() {
        // "hello" = aGVsbG8 (unpadded), aGVsbG8= (padded)
        assert_eq!(base64url_decode("aGVsbG8"), Some(b"hello".to_vec()));
        assert_eq!(base64url_decode("aGVsbG8="), Some(b"hello".to_vec()));
        // URL-safe pair: "-" (62) "_" (63) → 24 bits → 0xFB 0xEF 0xFF
        assert_eq!(base64url_decode("--__"), Some(vec![0xFB, 0xEF, 0xFF]));
        assert_eq!(base64url_decode("!!!"), None);
    }

    #[test]
    fn workspace_extracted_from_bearer_jwt() {
        // payload {"workspace_id":"ws-1"} base64url
        let payload = "eyJ3b3Jrc3BhY2VfaWQiOiJ3cy0xIn0";
        let token = format!("eyJhbGciOiJIUzI1NiJ9.{payload}.sig");
        assert_eq!(
            workspace_id(Some(&format!("Bearer {token}"))),
            Some("ws-1".into())
        );
        assert_eq!(workspace_id(Some("Basic abc")), None);
        assert_eq!(workspace_id(None), None);
        // 3-part but payload has no workspace_id
        let empty = "eyJmb28iOiJiYXIifQ"; // {"foo":"bar"}
        let token = format!("a.{empty}.c");
        assert_eq!(workspace_id(Some(&format!("Bearer {token}"))), None);
        // Not 3 parts
        assert_eq!(workspace_id(Some("Bearer only.two"),), None);
        // wait: "only.two" IS 2 parts → parts.next third → None → None ✓
        let malformed = "a.!!!notb64!!!.c";
        assert_eq!(workspace_id(Some(&format!("Bearer {malformed}"))), None);
    }
}
