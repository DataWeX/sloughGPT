//! Strict edge path filters — transport policy only, no model knowledge.
//!
//! Rejected paths never wake Python. Two layers:
//! 1. Always-on hygiene (null bytes, `..` traversal).
//! 2. Optional policy: deny-prefix list (`MAN_GATEWAY_DENY`) and
//!    chat-only mode (`MAN_GATEWAY_CHAT_ONLY=1`) which allowlists
//!    conversation-shaped prefixes and blocks everything else.

/// Path segments that must never reach the sidecar as-is.
const ALWAYS_DENY_SUBSTRINGS: [&str; 3] = ["\0", "/../", ".."];

/// Prefixes allowed when chat-only mode is on (conversation surface + docs).
/// Auth/rate-limits stay in Python — this is exposure shape, not security.
const CHAT_ONLY_ALLOW: [&str; 14] = [
    "/chat",
    "/inference",
    "/models",
    "/session",
    "/memory",
    "/companion",
    "/souls",
    "/feedback",
    "/conversations",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/system",
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PathPolicy {
    /// When true, only `CHAT_ONLY_ALLOW` prefixes pass (plus gateway `/health*`).
    pub chat_only: bool,
    /// Extra deny prefixes from `MAN_GATEWAY_DENY` (comma-separated).
    pub deny_prefixes: Vec<String>,
}

impl PathPolicy {
    pub fn from_env() -> Self {
        let chat_only = matches!(
            std::env::var("MAN_GATEWAY_CHAT_ONLY").as_deref(),
            Ok("1") | Ok("true") | Ok("yes") | Ok("on")
        );
        let deny_prefixes = std::env::var("MAN_GATEWAY_DENY")
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        if s.starts_with('/') {
                            s.to_string()
                        } else {
                            format!("/{s}")
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            chat_only,
            deny_prefixes,
        }
    }

    /// `Ok(())` when the path may be relayed; `Err(reason)` when blocked.
    pub fn check(&self, path: &str) -> Result<(), &'static str> {
        // Hygiene: never forward raw traversal / NUL.
        for bad in ALWAYS_DENY_SUBSTRINGS {
            if path.contains(bad) {
                return Err("path rejected by edge filter");
            }
        }
        // Explicit deny list wins first (same boundary rule as allow: `/shell` blocks `/shell/run`, not `/shelladmin`).
        for deny in &self.deny_prefixes {
            if path_starts_with(path, deny) {
                return Err("path denied by edge policy");
            }
        }
        // Gateway-owned health stays available even in chat-only mode.
        if path == "/health" || path.starts_with("/health/") {
            return Ok(());
        }
        if self.chat_only && !CHAT_ONLY_ALLOW.iter().any(|p| path_starts_with(path, p)) {
            return Err("chat-only edge: path not allowed");
        }
        Ok(())
    }
}

/// Prefix match on path boundaries: `/chat` allows `/chat` and `/chat/stream`,
/// but not `/chatterbox`.
fn path_starts_with(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .map(|rest| rest.starts_with('/') || rest.is_empty())
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(chat_only: bool, deny: &[&str]) -> PathPolicy {
        PathPolicy {
            chat_only,
            deny_prefixes: deny.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn blocks_traversal_and_null() {
        let p = policy(false, &[]);
        assert!(p.check("/models/../etc/passwd").is_err());
        assert!(p.check("/models\0x").is_err());
        assert!(p.check("/../secret").is_err());
        assert!(p.check("/models").is_ok());
    }

    #[test]
    fn deny_prefixes_block_family() {
        let p = policy(false, &["/shell", "/vm"]);
        assert!(p.check("/shell").is_err());
        assert!(p.check("/shell/run").is_err());
        assert!(p.check("/vm/console").is_err());
        assert!(p.check("/chat").is_ok());
    }

    #[test]
    fn chat_only_allows_conversation_surface() {
        let p = policy(true, &[]);
        for ok in [
            "/chat",
            "/chat/stream",
            "/inference/generate",
            "/models",
            "/session/list",
            "/openapi.json",
            "/docs",
            "/health",
            "/health/detailed",
        ] {
            assert!(p.check(ok).is_ok(), "{ok} should pass");
        }
        for bad in ["/shell", "/datasets", "/training/start", "/knowledge"] {
            assert!(p.check(bad).is_err(), "{bad} should block");
        }
    }

    #[test]
    fn chat_only_prefix_is_boundary_aware() {
        let p = policy(true, &[]);
        // "/chat" must not accidentally allow a different top-level name
        // that merely shares a string prefix without a slash boundary.
        assert!(p.check("/chatterbox").is_err());
        assert!(p.check("/chat").is_ok());
    }

    #[test]
    fn full_mode_allows_non_chat() {
        let p = policy(false, &[]);
        assert!(p.check("/datasets").is_ok());
        assert!(p.check("/shell").is_ok());
    }

    #[test]
    fn deny_wins_over_chat_allow() {
        let p = policy(true, &["/models"]);
        assert!(p.check("/models").is_err());
        assert!(p.check("/chat").is_ok());
    }

    #[test]
    fn path_starts_with_helpers() {
        assert!(path_starts_with("/chat", "/chat"));
        assert!(path_starts_with("/chat/stream", "/chat"));
        assert!(!path_starts_with("/chatter", "/chat"));
        assert!(!path_starts_with("/a/chat", "/chat"));
    }
}
