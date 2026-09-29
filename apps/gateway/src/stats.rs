//! Edge performance stats — what the app structurally cannot see.
//!
//! FastAPI's `/health/detailed` metrics only observe traffic that *reached*
//! Python; this module observes the full edge picture: relayed latency and
//! bytes, compression savings, and everything the edge shed (429/503) before
//! Python woke. The gateway reads its own measurements — independent of the
//! backend measuring a process too saturated to measure itself.
//!
//! Bounded by construction: routes keyed by first path segment, capped at
//! 32 slots + `(other)` overflow, fixed-size latency histogram, one mutex
//! held for counter arithmetic only (never across `.await`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Distinct route slots before aggregation into `(other)`.
const MAX_ROUTES: usize = 32;
/// Overflow bucket name.
const OTHER: &str = "(other)";
/// Log2-µs buckets: 0 = ≤1µs … 25 = 16–33.5s (covers BUFFERED_TIMEOUT=30s).
const HIST_BUCKETS: usize = 26;

/// First path segment as the stats key: `/chat/stream` → `/chat`.
/// Collapses UUID/ID paths (`/api/v1/abc-123` → `/api`) so clients cannot
/// grow the table past the cap.
pub fn route_key(path: &str) -> String {
    let rest = path.strip_prefix('/').unwrap_or(path);
    match rest.split('/').next() {
        Some("") | None => "/".to_string(),
        Some(first) => format!("/{first}"),
    }
}

#[derive(Default, Clone)]
pub struct RouteStat {
    pub requests: u64,
    pub ok_2xx: u64,
    pub err_4xx: u64,
    pub err_5xx: u64,
    pub shed_429: u64,
    pub shed_503: u64,
    /// Transport failures / edge timeouts to the core (connect refused,
    /// timeout, reset) — the failures the breaker acts on.
    pub upstream_err: u64,
    pub bytes_in: u64,
    /// Identity (pre-compression) response bytes.
    pub bytes_identity: u64,
    /// What actually went over the wire (post-compression).
    pub bytes_wire: u64,
    /// Latency-to-response-headers histogram, log2 µs.
    pub latency_hist: [u64; HIST_BUCKETS],
    pub latency_count: u64,
}

impl RouteStat {
    fn record_latency(&mut self, d: Duration) {
        let us = d.as_micros().min(u128::from(u64::MAX)) as u64;
        let b = if us == 0 {
            0
        } else {
            (63 - us.leading_zeros()) as usize
        };
        let b = b.min(HIST_BUCKETS - 1);
        self.latency_hist[b] += 1;
        self.latency_count += 1;
    }

    /// Approximate percentile in milliseconds (histogram midpoint).
    fn pct_ms(&self, p: f64) -> f64 {
        if self.latency_count == 0 {
            return 0.0;
        }
        let target = (self.latency_count as f64 * p).ceil() as u64;
        let mut acc = 0u64;
        for (b, &c) in self.latency_hist.iter().enumerate() {
            acc += c;
            if acc >= target {
                let lo = (1u64 << b) as f64;
                let hi = (1u64 << (b + 1)) as f64;
                return ((lo + hi) / 2.0) / 1000.0;
            }
        }
        0.0
    }
}

struct Inner {
    routes: Vec<(String, RouteStat)>,
}

pub struct EdgeStats {
    inner: Mutex<Inner>,
    in_flight: AtomicUsize,
}

impl EdgeStats {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner { routes: Vec::new() }),
            in_flight: AtomicUsize::new(0),
        }
    }

    fn slot<'a>(inner: &'a mut Inner, key: &str) -> &'a mut RouteStat {
        if let Some(i) = inner.routes.iter().position(|(k, _)| k == key) {
            return &mut inner.routes[i].1;
        }
        if inner.routes.len() < MAX_ROUTES {
            inner.routes.push((key.to_string(), RouteStat::default()));
            return &mut inner.routes.last_mut().expect("just pushed").1;
        }
        if let Some(i) = inner.routes.iter().position(|(k, _)| k == OTHER) {
            return &mut inner.routes[i].1;
        }
        inner.routes.push((OTHER.to_string(), RouteStat::default()));
        &mut inner.routes.last_mut().expect("just pushed").1
    }

    fn with_route(&self, route: &str, f: impl FnOnce(&mut RouteStat)) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        f(Self::slot(&mut inner, route));
    }

    /// Successful upstream exchange: status class + header latency.
    pub fn record_upstream(&self, route: &str, status: u16, latency: Duration) {
        self.with_route(route, |r| {
            r.requests += 1;
            match status {
                0..=399 => r.ok_2xx += 1,
                400..=499 => r.err_4xx += 1,
                _ => r.err_5xx += 1,
            }
            r.record_latency(latency);
        });
    }

    pub fn add_bytes_in(&self, route: &str, n: usize) {
        self.with_route(route, |r| r.bytes_in += n as u64);
    }

    pub fn add_identity(&self, route: &str, n: usize) {
        self.with_route(route, |r| r.bytes_identity += n as u64);
    }

    pub fn add_wire(&self, route: &str, n: usize) {
        self.with_route(route, |r| r.bytes_wire += n as u64);
    }

    pub fn shed_429(&self, route: &str) {
        self.with_route(route, |r| r.shed_429 += 1);
    }

    pub fn shed_503(&self, route: &str) {
        self.with_route(route, |r| r.shed_503 += 1);
    }

    pub fn add_upstream_err(&self, route: &str) {
        self.with_route(route, |r| r.upstream_err += 1);
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    fn in_flight_inc(&self) {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    fn in_flight_dec(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    /// Aggregate totals + per-route rows (latency p50/p95 from histogram).
    pub fn snapshot(&self) -> serde_json::Value {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut totals = serde_json::json!({
            "requests": 0u64, "2xx": 0u64, "4xx": 0u64, "5xx": 0u64,
            "shed_429": 0u64, "shed_503": 0u64, "upstream_err": 0u64,
            "bytes_in": 0u64, "bytes_identity": 0u64, "bytes_wire": 0u64,
        });
        let mut rows = Vec::with_capacity(inner.routes.len());
        for (key, r) in &inner.routes {
            totals["requests"] = json_u64_add(&totals["requests"], r.requests);
            totals["2xx"] = json_u64_add(&totals["2xx"], r.ok_2xx);
            totals["4xx"] = json_u64_add(&totals["4xx"], r.err_4xx);
            totals["5xx"] = json_u64_add(&totals["5xx"], r.err_5xx);
            totals["shed_429"] = json_u64_add(&totals["shed_429"], r.shed_429);
            totals["shed_503"] = json_u64_add(&totals["shed_503"], r.shed_503);
            totals["upstream_err"] = json_u64_add(&totals["upstream_err"], r.upstream_err);
            totals["bytes_in"] = json_u64_add(&totals["bytes_in"], r.bytes_in);
            totals["bytes_identity"] = json_u64_add(&totals["bytes_identity"], r.bytes_identity);
            totals["bytes_wire"] = json_u64_add(&totals["bytes_wire"], r.bytes_wire);
            let saved = r.bytes_identity.saturating_sub(r.bytes_wire);
            rows.push(serde_json::json!({
                "route": key,
                "requests": r.requests,
                "2xx": r.ok_2xx,
                "4xx": r.err_4xx,
                "5xx": r.err_5xx,
                "shed_429": r.shed_429,
                "shed_503": r.shed_503,
                "upstream_err": r.upstream_err,
                "p50_ms": round1(r.pct_ms(0.50)),
                "p95_ms": round1(r.pct_ms(0.95)),
                "bytes_in": r.bytes_in,
                "bytes_identity": r.bytes_identity,
                "bytes_wire": r.bytes_wire,
                "compression_saved": saved,
            }));
        }
        serde_json::json!({ "totals": totals, "routes": rows })
    }
}

/// Guard that keeps the in-flight gauge honest: dropped when the stream body
/// finishes, aborts, or the request errors anywhere downstream.
pub fn stream_open(stats: &Arc<EdgeStats>) -> StreamGuard {
    stats.in_flight_inc();
    StreamGuard(stats.clone())
}

pub struct StreamGuard(Arc<EdgeStats>);

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.0.in_flight_dec();
    }
}

fn json_u64_add(v: &serde_json::Value, n: u64) -> serde_json::Value {
    serde_json::json!(v.as_u64().unwrap_or(0) + n)
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_key_is_first_segment_and_collapses_ids() {
        assert_eq!(route_key("/chat/stream"), "/chat");
        assert_eq!(route_key("/api/v1/deadbeef-1234/messages"), "/api");
        assert_eq!(route_key("/models/load"), "/models");
        assert_eq!(route_key("/"), "/");
        assert_eq!(route_key("no-slash"), "/no-slash");
    }

    #[test]
    fn records_latency_status_and_bytes() {
        let s = EdgeStats::new();
        s.record_upstream("/chat", 200, Duration::from_millis(50));
        s.record_upstream("/chat", 503, Duration::from_secs(1));
        s.add_bytes_in("/chat", 100);
        s.add_identity("/chat", 1000);
        s.add_wire("/chat", 300);
        let snap = s.snapshot();
        let route = &snap["routes"][0];
        assert_eq!(route["requests"], 2);
        assert_eq!(route["2xx"], 1);
        assert_eq!(route["5xx"], 1);
        assert_eq!(route["bytes_identity"], 1000);
        assert_eq!(route["compression_saved"], 700);
        assert!(route["p50_ms"].as_f64().unwrap() > 0.0);
        assert_eq!(snap["totals"]["requests"], 2);
    }

    #[test]
    fn percentile_lands_in_expected_bucket() {
        let mut r = RouteStat::default();
        for _ in 0..90 {
            r.record_latency(Duration::from_micros(500)); // ~0.5ms
        }
        for _ in 0..10 {
            r.record_latency(Duration::from_millis(20));
        }
        let p50 = r.pct_ms(0.50);
        let p95 = r.pct_ms(0.95);
        // p50 inside the 256–512µs bucket (midpoint 0.384ms), p95 inside 16–32ms.
        assert!((0.2..0.6).contains(&p50), "p50={p50}");
        assert!((16.0..32.0).contains(&p95), "p95={p95}");
    }

    #[test]
    fn cardinality_capped_with_other_overflow() {
        let s = EdgeStats::new();
        for i in 0..40 {
            s.record_upstream(&format!("/r{i}"), 200, Duration::from_millis(1));
        }
        let snap = s.snapshot();
        let rows = snap["routes"].as_array().unwrap();
        assert_eq!(rows.len(), MAX_ROUTES + 1, "cap + (other)");
        let other = rows.iter().find(|r| r["route"] == "(other)").unwrap();
        assert_eq!(other["requests"], 40 - MAX_ROUTES as u64);
    }

    #[test]
    fn shed_counters_count() {
        let s = EdgeStats::new();
        s.shed_429("/chat");
        s.shed_429("/chat");
        s.shed_503("/chat");
        let snap = s.snapshot();
        assert_eq!(snap["routes"][0]["shed_429"], 2);
        assert_eq!(snap["routes"][0]["shed_503"], 1);
        assert_eq!(snap["totals"]["shed_429"], 2);
    }

    #[test]
    fn stream_guard_tracks_in_flight() {
        let s = Arc::new(EdgeStats::new());
        assert_eq!(s.in_flight(), 0);
        let g1 = stream_open(&s);
        let g2 = stream_open(&s);
        assert_eq!(s.in_flight(), 2);
        drop(g1);
        assert_eq!(s.in_flight(), 1);
        drop(g2);
        assert_eq!(s.in_flight(), 0);
    }
}
