---
id: 20261001_gateway_binary_provenance
title: "Gateway binary provenance: built from foreign worktree, stale"
status: done
date: 2026-10-01
tags: [gateway, build, provenance, rust]
board_id: 12aad775-3327-41c0-9fa8-e497145432d6
board: 12aad775-3327-41c0-9fa8-e497145432d6
hash: 6ab01f57f938734c06a35159d79b8121d3c51dafcab10a48539846f00d007151
---

# Gateway binary provenance: built from foreign worktree, stale

apps/gateway/target/release/slough-gateway (Sep 28 12:20) has .d stamps from
/home/mana/Documents/sloughGPT-edge-cascade (worktree, branch
feat/gateway-edge-cascade), copied into this checkout's target/. That branch has
commits through Sep 30 → binary stale vs even its own source; running unit
slough-gateway.service (PID then 1646) serves that build's /health passthrough
instead of this checkout's ~200-byte HealthResponse. Acceptance: one source of
truth — rebuild with cargo build --release from the shared checkout (or document
the edge-cascade build as canonical and sync it), systemctl --user restart
slough-gateway, :8080/health matches the binary's contract.

SHARED WORKSPACE
- index updated: no index change required (no new app/package/doc)
- shared copy reused: rebuild into this checkout's target/ — no new toolchain
- coordination named: source worktree feat/gateway-edge-cascade
  (/home/mana/Documents/sloughGPT-edge-cascade) is the provenance reference, not
  a live owner of this fix


[CLAIMED+DONE 2026-10-20 (.wt-worldbrain session)] Found PARTIALLY remediated
already (binary rebuilt Oct 19 14:33, 0 foreign .d stamps, service restarted
09:07) but still stale vs its own source: `cargo build --release --locked -p
slough-gateway` in the shared checkout recompiled the gateway crate (1m10s) ->
new binary Oct 20 10:50 (8.4MB). `systemctl --user restart slough-gateway` ->
active 10:50:49, PID 171770. :8080/health = {"status":"ok","gateway":"rust",
"sidecar":{healthy,last_check,model_loaded,model_name},"uptime_seconds":3} —
field-for-field the shared checkout's HealthResponse (src/main.rs:146, handler
:380; uptime=3s proves the fresh process serves it). Provenance: `strings binary
| grep -c sloughGPT-edge-cascade` = 0; foreign .d stamps = 0. One source of
truth: the unit's WorkingDirectory/ExecStart already point at the shared
checkout (no unit edits); edge-cascade worktree stays the provenance reference
only. No source changes (target/ ignored) — closes as verify+rebuild+restart
evidence. Note hand-edited: PlannerStore cannot see legacy .md journal files
(reads .dev-notes/notes.journal.jsonl only) — CLI note update unreachable; the
`hash:` frontmatter field is vestigial (no code reads it).