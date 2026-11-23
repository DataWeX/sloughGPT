//! Circuit breaker — the first stats-driven actuator.
//!
//! Closed → Open (K consecutive upstream failures) → half-open probe after
//! the open window → Closed on success. While open, requests shed instantly
//! with `503 {"detail"}` + `Retry-After` instead of waiting out edge timeouts
//! against a dead core. Fed only by **edge-observed request outcomes** —
//! health-poll observations are deliberately excluded so a healthy poll can
//! never mask a run of failing requests (no false reset).

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Consecutive failures to trip. `0` disables the breaker.
/// Env: `MAN_GATEWAY_BREAKER_FAILURES` (default 5).
/// Open window seconds. Env: `MAN_GATEWAY_BREAKER_OPEN` (default 5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakerConfig {
    pub failures: usize,
    pub open_secs: u64,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failures: 5,
            open_secs: 5,
        }
    }
}

impl BreakerConfig {
    pub fn from_env() -> Self {
        Self::parse(
            std::env::var("MAN_GATEWAY_BREAKER_FAILURES")
                .ok()
                .as_deref(),
            std::env::var("MAN_GATEWAY_BREAKER_OPEN").ok().as_deref(),
        )
    }

    /// Unset or unparseable → default. `"0"` disables.
    pub fn parse(failures: Option<&str>, open: Option<&str>) -> Self {
        let d = Self::default();
        Self {
            failures: failures
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(d.failures),
            open_secs: open
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(d.open_secs),
        }
    }

    pub fn enabled(&self) -> bool {
        self.failures > 0 && self.open_secs > 0
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum State {
    Closed {
        consecutive: usize,
    },
    Open {
        until: Instant,
    },
    /// Exactly one probe allowed through; others wait.
    HalfOpen,
}

pub struct CircuitBreaker {
    cfg: BreakerConfig,
    state: Mutex<State>,
}

impl CircuitBreaker {
    pub fn new(cfg: BreakerConfig) -> Self {
        Self {
            cfg,
            state: Mutex::new(State::Closed { consecutive: 0 }),
        }
    }

    /// `Ok(())` lets the request through; `Err(retry_after_secs)` sheds it.
    pub fn allow(&self, now: Instant) -> Result<(), u64> {
        if !self.cfg.enabled() {
            return Ok(());
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match *st {
            State::Closed { .. } => Ok(()),
            State::Open { until } => match until.checked_duration_since(now) {
                // Window elapsed → admit exactly one probe.
                None | Some(Duration::ZERO) => {
                    *st = State::HalfOpen;
                    Ok(())
                }
                Some(remaining) => Err(remaining.as_secs().max(1)),
            },
            // A probe is already in flight.
            State::HalfOpen => Err(1),
        }
    }

    /// Upstream answered (any status): success unless 5xx.
    pub fn observe(&self, status: u16) {
        if status >= 500 {
            self.record_failure();
        } else {
            self.record_success();
        }
    }

    pub fn record_success(&self) {
        if !self.cfg.enabled() {
            return;
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match *st {
            // Ignore while open: only the half-open probe may close it, so
            // late successes from requests admitted before the trip can't
            // mask an ongoing failure run.
            State::Closed { .. } => *st = State::Closed { consecutive: 0 },
            State::HalfOpen => *st = State::Closed { consecutive: 0 },
            State::Open { .. } => {}
        }
    }

    pub fn record_failure(&self) {
        if !self.cfg.enabled() {
            return;
        }
        let now = Instant::now();
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match *st {
            State::Closed { consecutive } => {
                let consecutive = consecutive + 1;
                if consecutive >= self.cfg.failures {
                    *st = State::Open {
                        until: now + Duration::from_secs(self.cfg.open_secs),
                    };
                } else {
                    *st = State::Closed { consecutive };
                }
            }
            // Probe failed → straight back to open (fresh window).
            State::HalfOpen => {
                *st = State::Open {
                    until: now + Duration::from_secs(self.cfg.open_secs),
                }
            }
            State::Open { .. } => {}
        }
    }

    /// `(state_name, retry_after_secs_if_shed)` for stats/health output.
    pub fn status(&self, now: Instant) -> (&'static str, u64) {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match *st {
            State::Closed { .. } => ("closed", 0),
            State::HalfOpen => ("half_open", 1),
            State::Open { until } => {
                let retry = until
                    .checked_duration_since(now)
                    .map(|d| d.as_secs().max(1))
                    .unwrap_or(0);
                ("open", retry)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breaker(failures: usize, open_secs: u64) -> CircuitBreaker {
        CircuitBreaker::new(BreakerConfig {
            failures,
            open_secs,
        })
    }

    #[test]
    fn parse_defaults_and_zero_disables() {
        assert_eq!(BreakerConfig::parse(None, None), BreakerConfig::default());
        let cfg = BreakerConfig::parse(Some("0"), Some("0"));
        assert!(!cfg.enabled());
        let cfg = BreakerConfig::parse(Some("abc"), Some("7"));
        assert_eq!(cfg.failures, BreakerConfig::default().failures);
        assert_eq!(cfg.open_secs, 7);
    }

    #[test]
    fn disabled_never_trips() {
        let b = breaker(0, 5);
        for _ in 0..100 {
            b.record_failure();
            assert_eq!(b.allow(Instant::now()), Ok(()));
        }
    }

    #[test]
    fn opens_after_k_consecutive_failures() {
        let b = breaker(3, 60);
        let now = Instant::now();
        b.record_failure();
        b.record_failure();
        assert_eq!(b.allow(now), Ok(()), "2 of 3 → still closed");
        b.record_failure();
        assert!(b.allow(now).is_err(), "3rd failure → open");
        let (state, retry) = b.status(now);
        assert_eq!(state, "open");
        assert!(retry >= 1);
    }

    #[test]
    fn success_resets_consecutive_run() {
        let b = breaker(3, 60);
        b.record_failure();
        b.record_failure();
        b.record_success();
        b.record_failure();
        b.record_failure();
        assert_eq!(b.allow(Instant::now()), Ok(()), "run was reset by success");
    }

    #[test]
    fn late_success_while_open_cannot_close_it() {
        let b = breaker(2, 60);
        b.record_failure();
        b.record_failure();
        assert!(b.allow(Instant::now()).is_err());
        b.record_success(); // stale in-flight success — must be ignored
        let (state, _) = b.status(Instant::now());
        assert_eq!(state, "open");
    }

    #[test]
    fn half_open_admits_one_probe_then_reopens_on_failure() {
        half_open_state_machine();
    }

    /// Direct state-machine test for probe semantics (no sleeping).
    fn half_open_state_machine() {
        let cfg = BreakerConfig {
            failures: 1,
            open_secs: 1,
        };
        let b = CircuitBreaker::new(cfg);
        b.record_failure();
        let now = Instant::now();
        assert!(b.allow(now).is_err());
        let later = now + Duration::from_secs(2);
        // Window elapsed → first allow becomes the probe.
        assert_eq!(b.allow(later), Ok(()));
        // Second concurrent caller must wait for the probe result.
        assert!(b.allow(later).is_err());
        // Probe succeeds → closed.
        b.record_success();
        assert_eq!(b.allow(later), Ok(()));
        // Re-trip: the fresh window is anchored at the *real* clock
        // (record_failure uses Instant::now()), so probe against `now`
        // (inside the window → shed) and `later` (past it → admit).
        b.record_failure();
        assert!(b.allow(now).is_err());
        assert_eq!(b.allow(later), Ok(())); // probe
        b.record_failure();
        let (state, retry) = b.status(now);
        assert_eq!(state, "open");
        assert!(retry >= 1);
    }

    #[test]
    fn observe_maps_status_classes() {
        let b = breaker(3, 60);
        b.record_failure();
        b.record_failure();
        b.observe(404); // client errors are successes → run resets
        assert_eq!(b.allow(Instant::now()), Ok(()));
        b.observe(500); // 1st server error
        b.observe(502); // 2nd
        assert_eq!(b.allow(Instant::now()), Ok(()), "2 of 3 → still closed");
        b.observe(503); // 3rd → trips
        assert!(b.allow(Instant::now()).is_err());
    }
}
