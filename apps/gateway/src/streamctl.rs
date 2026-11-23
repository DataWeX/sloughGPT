//! Adaptive stream concurrency — the gateway *manages* inference capacity.
//!
//! AIMD (additive increase / multiplicative decrease, the TCP congestion
//! control idea applied to edge admission): the fixed `MAN_GATEWAY_MAX_STREAMS`
//! cap becomes a ceiling; the *effective* limit shrinks when admitted streams
//! fail against the core (transport errors, ≥500 relays) and grows back one
//! slot per healthy window. Floor is 1 — always leave one probe flowing,
//! same philosophy as the circuit breaker's half-open probe.
//!
//! Window: evaluated at most every [`TUNE_INTERVAL`], fed by counters reset
//! on evaluation. `failures / starts ≥ 1/DECREASE` → halve; otherwise
//! healthy → +1 up to max. Admission itself is a CAS on the stats in-flight
//! gauge (same population — see `stats::try_stream_slot`), so the gauge and
//! the cap can never disagree.
//!
//! Complements, never replaces, the circuit breaker: the breaker handles
//! dead-core (fail fast, everything sheds); StreamControl handles the middle
//! band — core alive but struggling — where full shutdown would be wrong.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Minimum spacing between limit evaluations.
const TUNE_INTERVAL: Duration = Duration::from_secs(2);
/// Failure ratio threshold: `failures * DECREASE ≥ starts` (≥ 12.5%)
/// in a window halves the limit.
const DECREASE: usize = 8;

pub struct StreamControl {
    /// Configured ceiling (`MAN_GATEWAY_MAX_STREAMS`).
    max: usize,
    /// Effective admission limit — what `try_stream_slot` CASes against.
    limit: AtomicUsize,
    /// Admitted streams since last tune.
    starts: AtomicUsize,
    /// Stream failures since last tune (transport/timeout/≥500 relay).
    failures: AtomicUsize,
    last_tune: Mutex<Instant>,
}

impl StreamControl {
    /// `max` comes from `MAN_GATEWAY_MAX_STREAMS`; callers gate on `> 0`
    /// (0 = disabled), but the struct stays sane either way (floor 1).
    pub fn new(max: usize) -> Self {
        let effective_max = max.max(1);
        Self {
            max: effective_max,
            limit: AtomicUsize::new(effective_max),
            starts: AtomicUsize::new(0),
            failures: AtomicUsize::new(0),
            last_tune: Mutex::new(Instant::now()),
        }
    }

    /// Current effective limit — admission CAS bound; surfaced in
    /// `/gateway/stats` as `stream_limit`.
    pub fn limit(&self) -> usize {
        self.limit.load(Ordering::Relaxed)
    }

    pub fn max(&self) -> usize {
        self.max
    }

    /// One admitted stream — feeds the window's start count.
    pub fn record_start(&self) {
        self.starts.fetch_add(1, Ordering::Relaxed);
    }

    /// One stream failure against the core (transport error, edge timeout,
    /// or a ≥500 relayed on a streaming path).
    pub fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Evaluate the window and adjust the effective limit. Called at stream
    /// admission (the only place capacity matters); rate-limited by
    /// [`TUNE_INTERVAL`]. `now` is injectable for tests.
    pub fn tune(&self, now: Instant) {
        let mut last = self.last_tune.lock().unwrap_or_else(|e| e.into_inner());
        if now.duration_since(*last) < TUNE_INTERVAL {
            return;
        }
        *last = now;
        let starts = self.starts.swap(0, Ordering::Relaxed);
        let failures = self.failures.swap(0, Ordering::Relaxed);
        if starts == 0 {
            return;
        }
        let cur = self.limit.load(Ordering::Relaxed);
        if failures > 0 && failures.saturating_mul(DECREASE) >= starts {
            self.limit.store((cur / 2).max(1), Ordering::Relaxed);
        } else if cur < self.max {
            self.limit.store((cur + 1).min(self.max), Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn later(t0: Instant, secs: u64) -> Instant {
        t0 + Duration::from_secs(secs)
    }

    #[test]
    fn halves_after_bad_window() {
        let ctl = StreamControl::new(32);
        let t0 = Instant::now();
        for _ in 0..8 {
            ctl.record_start();
        }
        ctl.record_failure(); // 1/8 = 12.5% → at threshold → halve
        ctl.tune(later(t0, 2));
        assert_eq!(ctl.limit(), 16);
    }

    #[test]
    fn healthy_window_grows_by_one_up_to_max() {
        let ctl = StreamControl::new(32);
        let t0 = Instant::now();
        // Halve first: 8 starts, 4 failures (50% ≥ 12.5%).
        for _ in 0..8 {
            ctl.record_start();
        }
        for _ in 0..4 {
            ctl.record_failure();
        }
        ctl.tune(later(t0, 2));
        assert_eq!(ctl.limit(), 16);
        // Healthy window → additive increase.
        for _ in 0..3 {
            ctl.record_start();
        }
        ctl.tune(later(t0, 4));
        assert_eq!(ctl.limit(), 17);
    }

    #[test]
    fn limit_never_drops_below_one() {
        let ctl = StreamControl::new(4);
        let t0 = Instant::now();
        for step in 1..=6u64 {
            ctl.record_start();
            ctl.record_failure();
            ctl.tune(later(t0, 2 * step));
        }
        assert_eq!(ctl.limit(), 1, "floor is one probe, never zero");
    }

    #[test]
    fn no_tune_before_interval_elapses() {
        let ctl = StreamControl::new(32);
        let t0 = Instant::now();
        for _ in 0..8 {
            ctl.record_start();
            ctl.record_failure();
        }
        ctl.tune(later(t0, 1)); // 1s < 2s → ignored, window kept
        assert_eq!(ctl.limit(), 32);
        ctl.tune(later(t0, 2));
        assert_eq!(ctl.limit(), 16, "window still had the earlier failures");
    }

    #[test]
    fn empty_window_changes_nothing() {
        let ctl = StreamControl::new(32);
        let t0 = Instant::now();
        ctl.tune(later(t0, 3));
        assert_eq!(ctl.limit(), 32);
    }

    #[test]
    fn repeated_healthy_windows_reach_max_and_stop() {
        let ctl = StreamControl::new(4);
        let t0 = Instant::now();
        // Force a halve, then grow back.
        for _ in 0..8 {
            ctl.record_start();
            ctl.record_failure();
        }
        ctl.tune(later(t0, 2));
        assert_eq!(ctl.limit(), 2);
        for step in 2..=8u64 {
            ctl.record_start();
            ctl.tune(later(t0, 2 * step));
        }
        assert_eq!(ctl.limit(), 4, "capped at max, never above");
    }
}
