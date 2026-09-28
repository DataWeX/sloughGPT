//! Self-supervision — the binary outlives its own crashes with zero OS
//! coupling (no systemd, no init scripts).
//!
//! `MAN_GATEWAY_SUPERVISE=1` runs a parent loop that spawns
//! `current_exe()` as a worker (`MAN_GATEWAY_WORKER=1`) and respawns it
//! on abnormal exit with capped backoff. SIGTERM/SIGINT to the parent
//! chain-stops the worker (SIGTERM → the worker's graceful shutdown).
//! If the *parent* dies hard, the orphaned worker keeps serving — the
//! socket never drops for a supervisor crash.

use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::process::{Child, Command};
use tracing::{error, info, warn};

/// What this process should do, decided from its two mode flags.
#[derive(Debug, PartialEq, Eq)]
pub enum Mode {
    /// Plain server (default, and always when the worker flag is set —
    /// a worker must never re-enter supervision even if it inherited
    /// `MAN_GATEWAY_SUPERVISE` from its parent).
    Worker,
    /// Parent supervision loop.
    Supervisor,
}

pub fn decide(worker_flag: bool, supervise_flag: bool) -> Mode {
    if worker_flag {
        Mode::Worker
    } else if supervise_flag {
        Mode::Supervisor
    } else {
        Mode::Worker
    }
}

pub fn mode_from_env() -> Mode {
    decide(
        std::env::var_os("MAN_GATEWAY_WORKER").is_some(),
        std::env::var_os("MAN_GATEWAY_SUPERVISE").is_some(),
    )
}

/// Exponential backoff: 1s doubling, capped at 30s.
pub fn backoff_secs(consecutive_failures: u32) -> u64 {
    let shift = consecutive_failures.saturating_sub(1).min(5);
    (1u64 << shift).min(30)
}

/// A run this long counts as healthy — the failure counter resets.
const HEALTHY_RUN: Duration = Duration::from_secs(60);

/// Resolves on SIGTERM/SIGINT (pending forever elsewhere). Registering
/// this handler also makes tokio's signal driver own those signals, so
/// every await in the server/supervisor must stay wrapped in a select
/// that actually exits on it.
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        let mut int = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("install SIGINT handler");
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        std::future::pending::<()>().await;
    }
}

async fn terminate(child: &mut Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            // Graceful: the worker's serve loop watches SIGTERM.
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
            return;
        }
    }
    let _ = child.start_kill();
}

/// The supervision loop. Never returns an error — it logs and retries.
pub async fn run() {
    info!("supervisor: up — respawns worker on crash, SIGTERM chain-stops");
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            error!("supervisor: cannot resolve own executable: {e}");
            return;
        }
    };

    let mut failures: u32 = 0;
    loop {
        let mut child = match Command::new(&exe)
            .env("MAN_GATEWAY_WORKER", "1")
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let wait = backoff_secs(failures.saturating_add(1));
                error!("supervisor: spawn failed: {e}; retry in {wait}s");
                failures = failures.saturating_add(1);
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(wait)) => {}
                    _ = shutdown_signal() => return,
                }
                continue;
            }
        };

        let started = Instant::now();
        tokio::select! {
            status = child.wait() => {
                if matches!(&status, Ok(s) if s.success()) {
                    info!("supervisor: worker exited cleanly — stopping");
                    return;
                }
                let why = status
                    .map(|s| s.to_string())
                    .unwrap_or_else(|e| e.to_string());
                if started.elapsed() >= HEALTHY_RUN {
                    failures = 0;
                }
                failures = failures.saturating_add(1);
                let wait = backoff_secs(failures);
                warn!(
                    "supervisor: worker died ({why}); restart in {wait}s (failure #{failures})"
                );
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(wait)) => {}
                    _ = shutdown_signal() => return,
                }
            }
            _ = shutdown_signal() => {
                info!("supervisor: shutdown requested — stopping worker");
                terminate(&mut child).await;
                let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_flag_wins_over_inherited_supervise_flag() {
        // A worker inherits MAN_GATEWAY_SUPERVISE from its parent —
        // it must never re-enter the loop (that would fork-bomb).
        assert_eq!(decide(true, true), Mode::Worker);
        assert_eq!(decide(true, false), Mode::Worker);
        assert_eq!(decide(false, true), Mode::Supervisor);
        assert_eq!(decide(false, false), Mode::Worker);
    }

    #[test]
    fn backoff_doubles_and_caps_at_30() {
        assert_eq!(backoff_secs(0), 1);
        assert_eq!(backoff_secs(1), 1);
        assert_eq!(backoff_secs(2), 2);
        assert_eq!(backoff_secs(3), 4);
        assert_eq!(backoff_secs(4), 8);
        assert_eq!(backoff_secs(5), 16);
        assert_eq!(backoff_secs(6), 30);
        assert_eq!(backoff_secs(50), 30);
    }
}
