---
id: 20261115_101945_avion-live-state-widget-floating-side-panel-projecting-backe
status: todo
board: 20261115_101945_avion-live-state-widget-floating-side-panel-projecting-backe
title: "avion: live-state widget — floating side panel projecting backend match/provenance state, zero endpoints"
tags: ["avion", "ux", "widget"]
sprint: ""
gh: null
---

# avion: live-state widget — floating side panel projecting backend match/provenance state, zero endpoints

Follow-up to PR #338 (multi-match click). In-page floating widget showing what avion sees RIGHT NOW: match list with original-set ordinals, rects, hidden/skipped reasons, per-hit click results, recording state — the on-screen twin of the identify block. Design (user-approved 2026-11-15): (1) ZERO endpoints/server — pure projection of state the CLI already holds (endpoint rule: same process, function call, no transport; tier-b operator surface, never enters model tool registry); (2) NEW module avion/interact/widget.py — shadow-DOM overlay injected via backend.evaluate (picker.py injection path), render_state(frame) from a JSON-able dict; (3) push, don't poll — CLI pushes ONE state frame after _cmd_click/identify (WidgetBridge), page never re-runs matching (single source of truth, no drift); (4) observational invisibility: shadow DOM + zero light-DOM text (find_elements must probe 0 widget matches), fixed overlay pointer-events off, auto-hide during shot/recording captures; (5) SPA route changes keep it natively, full navigations re-inject (Page.frameNavigated or re-push after each action). Surface ladder: inline widget (cheapest rung), no page.tsx. Tests: widget-invisible-to-probes, state frame == identify block byte-for-byte, hide-on-shot.
