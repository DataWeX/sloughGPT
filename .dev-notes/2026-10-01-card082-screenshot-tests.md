---
id: 20261001_082
title: "UI screenshot tests: headers + basic interactions across pages"
status: done
sprint: 2026-10-01
tags: [testing, playwright, visual-regression, journeys]
gh: ""
created: 2026-10-01T05:55:00Z
---

# UI screenshot tests: headers + basic interactions across pages

Board card `ec2ae5c1` (ask 2026-08-30): "computer use in component tests of
basic UI interactions so we can screenshot them... run it on our headers
across pages". Playwright toHaveScreenshot/page.screenshot in journey tests;
none exist.

## Recon (2026-10-01)
- Playwright **1.62.0 already in shared `.venv`** + chromium-1234 cached in
  `~/.cache/ms-playwright` (launched: v151.0.7922.34) — **zero installs**.
- Journey tests live in `packages/core-py/tests/test_user_journeys.py`
  (555 lines, module fixtures, stale `BASE=:3000` = dev-stack.sh default;
  live stack serves `:5173` since the systemd card). No screenshots anywhere.
- Probe of live app: no auth bounce logged-out; app header
  (`Open menu | sloughGPT`) present on `/`, `/chat`, `/settings`,
  `/training`, `/shortcuts`; `/models` → `/developer` redirect (skip).
- Foreign staged core-py files exist (8) but NOT `test_user_journeys.py`;
  new file avoids them. Main tree held by OOP session (unpushed commits) →
  **worktree `.wt-shot`**, branch `feat/ui-screenshot-tests`.

## Do
- New `packages/core-py/tests/test_ui_screenshots.py` (marked slow+e2e →
  excluded from default `-m "not slow"` runs): header screenshots across
  pages + mobile menu interaction.
- Commit PNG baselines; prove teeth via tamper test (corrupt baseline →
  red → restore).

## Acceptance
- [ ] header screenshots across >=5 pages + >=1 interaction screenshot
- [ ] default pytest run still deselects the file
- [ ] tamper test: edited baseline fails, restored baseline passes
- [ ] zero installs (no pip/npm)

## Acceptance (2026-10-01) — all checked

- [x] header screenshots across >=5 pages + >=1 interaction screenshot
      (5 mobile-header baselines: / /chat /settings /training /shortcuts +
      open-drawer interaction baseline)
- [x] default pytest run still deselects the file (11 deselected)
- [x] tamper test: edited baseline fails (red + .actual/.diff written),
      restored baseline passes
- [x] zero installs (playwright 1.62 already in shared .venv; browser =
      flatpak chromium via export shim after the ms-playwright cache was
      wiped mid-session by an unknown actor; `playwright install` never run)

## Notes for future maintainers

- `expect().to_have_screenshot` does NOT exist in this playwright Python
  build (0 occurrences in the API) — `expect_snapshot()` does create-if-
  missing + byte-compare + PIL diff diagnostics instead.
- Desktop top bar is intentionally visibility-only: live status + time
  greeting make it pixel-unstable by design; /chat has no desktop topbar
  (immersive layout, verified live).
- Consciousness-status row in the drawer is LIVE model state → masked via
  `mask=[drawer.get_by_label("Consciousness status")]`.
- Run: `.venv/bin/python -m pytest -c pytest.ini
  packages/core-py/tests/test_ui_screenshots.py -m slow -v`
  (`-c pytest.ini` = root config registers the `e2e` marker; the nested
  packages/core-py/pytest.ini does not).

## Landed

- a9d3d079d — test(journeys): UI screenshot tests (branch
  feat/ui-screenshot-tests, worktree .wt-shot, pushed to
  feat/shared-governance)

Canonical note: 2026-09-29-card082-ui-screenshot-tests.md (id 20260929_082, status done) — this file is the extended work log.
