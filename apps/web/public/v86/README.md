# v86 emulator pair

- `libv86.js` + `v86.wasm` — copy.sh's current v86 demo build, fetched from
  https://copy.sh/v86/ on 2026-10-18 (served same-origin so the app has no
  runtime CDN dependency). This build is the one proven to boot the vendored
  Linux 6.8.12 kernel to the busybox shell prompt end-to-end in the Browser VM
  panel; it replaces the earlier npm `@copy/v86` 0.5.456 pair (kept in git
  history), whose screen adapter stalls identically but was never verified to
  a full shell.
- `v86-fallback.wasm` — earlier npm-package wasm kept from the original
  vendoring commit; unreferenced by default (`useV86` uses `/v86/v86.wasm`).

`public/v86-test.html` + `public/v86-test/` are a bare-metal boot harness
(same pair + same kernel + instrumentation) used to debug boot/render issues
without the app stack; see the Browser VM card on the kanban board.
