---
id: 20260930_100
title: "Run stack (api/gateway/vite) as systemd --user units, not agent-shell children"
status: done
date: 2026-09-30
tags: [infra, systemd, reliability, process-guard]
board_id: c488c312-e04c-4f23-ab4a-d76de4d5ba37
board: c488c312-e04c-4f23-ab4a-d76de4d5ba37
hash: ee78457dbe6a191f891cf68ac246a67d2f0f8bc5e4edcf629ffcb4061c69e50a
---

# Run stack (api/gateway/vite) as systemd --user units, not agent-shell children

## Background

Board card `c488c312` (created 2026-09-30 14:40): shell-owned processes get
cancelled on host/session restart — observed repeatedly. On 2026-09-30 alone the
opencode desktop restarts (3+) killed everything in their cgroup, **including
the shared `:8000` API server** (restored by hand mid-session as yet another
fragile shell child), plus vite `:5173`. Plan from the card: slough-api /
slough-gateway / slough-web units in `scripts/systemd/`, install to
`~/.config/systemd/user`, pattern = `packages/chargectl/systemd` +
`make charge-svc-user`, `Restart=on-failure`, `journalctl --user`. Open
questions: `loginctl enable-linger`? `enable --now` vs install-only? Register
in scripts README + docs/INDEX per shared-workspace rule.

## Expert review (of the card's proposed approach)

- Sound: systemd user units decouple service lifetime from the launching
  shell's cgroup — exactly the invariant that broke (cgroup-scoped kills take
  shell-children down). `Restart=on-failure` gives crash recovery that
  dev-stack.sh's watch-loop only provides while ITS parent shell lives.
- Right instincts: reuse the chargectl unit pattern + its install flow;
  journalctl logging; index registration.
- Refinement 1 (mechanism): units need absolute ExecStart + explicit
  `WorkingDirectory` + PATH (nvm node lives in `~/.nvm/versions/node/...`, not
  systemd's default PATH); install must be idempotent.
- Refinement 2 (scope): in-process probes/benchmarks unaffected (they never
  go through systemd); don't seize ports nobody supervises without takeover
  being explicit; `enable-linger` is machine config → flag-gated, not
  default.

## Facts gathered (2026-09-30)

- `scripts/dev-stack.sh` exists (286 lines): the canonical shell launcher
  (api+web, backoff restart, Ctrl+C cleanup) — it IS the agent-shell child
  pattern being replaced. API: `.venv/bin/python3 apps/api/server/main.py`
  (SLO_API_PORT default 8000); Web: `cd apps/web && npm run dev` → vite
  (live instance on :5173); Gateway: opt-in `MAN_DEV_GATEWAY=1`, binary
  `apps/gateway/target/release/slough-gateway` (EXISTS, 8.5MB, Sep 28),
  env `MAN_GATEWAY_PORT=8080 MAN_CORE_URL=http://127.0.0.1:8000`, /health.
- Live stack at pickup: api :8000 = shell child (restored 18:14 after restart
  killed original), vite :5173 = `npm run dev` child of a session, gateway
  off (dev-stack default).
- `~/.config/systemd/user/` has no stack units; `scripts/systemd/` absent.
- node/npm = nvm `~/.nvm/versions/node/v20.20.2/bin/`.
- api under conda python proven (card 093 temp :8010 server served 15
  benchmark requests).
- `scripts/README.md` does NOT exist → register where dev-stack is actually
  documented instead: root `README.md` quickstart (line ~29) +
  `docs/INDEX.md` (+ `docs/DEVELOPER_GUIDE.md` if it explains dev-stack).
  Deviation from card body recorded deliberately.

## Design decisions (recommendation-taking per standing directive)

1. Three units in `scripts/systemd/`: `slough-api.service`,
   `slough-web.service`, `slough-gateway.service`; `Restart=on-failure`,
   `RestartSec=5`, `Type=simple`.
2. api ExecStart = conda python (standing "conda not venv" directive,
   proven working), WorkingDirectory = MAIN repo root (units are
   machine-level config pointing at the canonical checkout, not a worktree).
3. web ExecStart = nvm node (absolute) running vite from
   `node_modules/.bin/vite` with cwd `apps/web`; explicit PATH for nvm.
4. gateway unit = binary + env as dev-stack does; enabled but only started
   if healthy (binary present ✓).
5. Install script `scripts/systemd/install.sh`: idempotent
   `install -m 644` → `~/.config/systemd/user`, `daemon-reload`;
   default = install + enable + restart services whose port is live or
   desired (takeover of shell children via explicit `--takeover`? see
   Evidence for what shipped); `--install-only` opt-out; `--linger`
   flag-gated (never default).
6. Docs: README quickstart + docs/INDEX registration (index rule).

## Gates

- [x] unit files pass `systemd-analyze verify` (caught a real bug: ExecStart
      splits on spaces → quoted the web/gateway paths; clean after)
- [x] install script idempotent (run twice, no errors)
- [x] api unit starts, /health 200 (+ /v1/models 200 — registry loads)
- [x] **restart-on-kill**: `kill -9` api → unit restarted it (NRestarts=1)
      → /health 200; same for web (:5173 → 200) and gateway (:8080 → 200)
- [x] web unit serves :5173, /chat returns 200 HTML
- [x] gateway unit /health 200 JSON on :8080
- [x] takeover: exactly one listener per port (8000/5173/8080)
- [x] `tests/test_shared_indexes.py` green (6/6) + docs registered
- [x] shared :8000 up at the end (other sessions depend on it) — now
      systemd-managed instead of a shell child
- [x] cgroup proof: units live in
      `user@1000.service/app.slice/<unit>.service`, NOT in the
      `app-ai.opencode.desktop-*.scope` where the shell children died —
      restart-proof by construction

## Evidence

### Progress log (in progress)

- **Deliverables** (`.wt-units`, branch `feat/systemd-units`, base
  cc6c13a81): `scripts/systemd/{slough-api,slough-web,slough-gateway}.service`
  + `scripts/systemd/install.sh` (flags: `--install-only`, `--takeover`,
  `--linger`; idempotent install → `~/.config/systemd/user` +
  `daemon-reload`; preflight skips gateway when its cargo binary is absent).
- **Key facts baked into units**: api = conda python (bare from repo root,
  health 200, no PYTHONPATH needed — verified on scratch :8012); web =
  absolute nvm node on `node_modules/.bin/vite` (systemd has no nvm init;
  matches the process npm spawns); gateway = release binary +
  `MAN_GATEWAY_PORT=8080 MAN_CORE_URL=http://127.0.0.1:8000` (dev-stack
  contract); `WorkingDirectory` with spaces parsed correctly (verified via
  `systemctl --user show -p WorkingDirectory`); `Restart=on-failure`,
  `RestartSec=5`, `TimeoutStopSec`, SIGTERM.
- **Migration (2026-09-30 22:53)**: `install.sh --takeover` took over
  :8000 from the restored shell child (pid 40161) and :5173 from the
  session's `npm run dev` child (pid 39283, its npm parent exits cleanly;
  URL unchanged), started gateway fresh. All three `active (running)` and
  `enabled` (auto-start at login; `loginctl enable-linger` deliberately
  NOT applied — flag-gated machine config, documented in --help).
- **Kill -9 gate**: api pid 103611 killed → restarted → /health 200;
  web and gateway likewise back to 200 within RestartSec. journalctl
  `--user -u slough-api` shows request logs.
- **Docs registered** (card said "scripts/README" — that file does not
  exist, so registered where dev-stack is actually documented, per the
  check-if-exists rule): root `README.md` quickstart block,
  `docs/INDEX.md` Setup & Deploy row, `docs/DEVELOPER_GUIDE.md`
  API+web section. `tests/test_shared_indexes.py` 6/6.
- **Scope held**: no in-process code touched (probe/benchmarks unaffected);
  no linger/boot-enable defaults flipped; dev-stack.sh left intact as the
  shell-based launcher; pre-existing stack processes replaced only via the
  explicit `--takeover` flag.

### Landed (2026-10-01, by a second session)

The original session died between its last gate and `git commit`; files sat
uncommitted in `.wt-units` for 7h. Landed on request (user-approved):
- `5ae57cb39` — deliverables committed byte-for-byte from `.wt-units` on
  `feat/systemd-units`, rebased onto the governance tip; gates re-run green
  (`systemd-analyze verify`, `install.sh --install-only` idempotence,
  shared-index pytest 6/6, :8000/:5173/:8080 all 200, units active).
- `d3017b204` — board done row. Pushed `96f7a71d9..d3017b204` to
  `feat/shared-governance`.
