---
id: 20260815_223512_workerts-di-handler-factory-last-viresetmodules-removed-from
title: worker.ts DI handler factory — last vi.resetModules removed from lib/
status: done
tags: web,tests,webgpu,worker,refactor
created: 2026-08-15T22:35:12.401212+00:00
---

worker.ts DI handler factory — last vi.resetModules removed from lib/

CONTEXT: user repeated 'code it properly' after the error-reporter/db DI round (committed 860d784d/93129e3f). The ONLY remaining vi.resetModules() in apps/web/lib was worker.test.ts:119.\n\nROOT CAUSE: worker.ts registered its message handler as an import-time side effect (self.onmessage = async (e) => {...}) with module-level 'let engine'. The 'errors on generate before init' test needed a fresh module so engine would be null again — hence vi.resetModules() + dynamic import + per-test vi.stubGlobal('self').\n\nFIX (apps/web/lib/soulnet-webgpu/worker.ts): extracted exported createWorkerHandler(workerScope: WorkerScope) which returns a per-instance async handler closing over its own 'let engine'; workerScope.postMessage replaces global self.postMessage. Exported registerWorker(workerScope = self) that installs createWorkerHandler(scope). Bottom-of-module guard 'if (typeof self !== undefined) registerWorker()' keeps the real bundler worker self-registering (index.ts:62 new Worker(new URL('./worker.ts', import.meta.url))). Production behavior identical.\n\nTESTS (worker.test.ts): static import of createWorkerHandler/registerWorker/WorkerScope; each test builds { postMessage: vi.fn(), onmessage: null } and awaits the handler directly (no vi.waitFor, no dynamic import, no self stub, no resetModules). Replaced 2 protocol tests + added 'does not share engine state across handlers' and 'registerWorker installs the handler on the given scope'. 8 worker tests, 36 soulnet-webgpu tests, lib/ 76 files / 867 tests all pass; npx tsc --noEmit exit 0. Committed 694786b7 (2 files, path-scoped).\n\nFINAL STATE: zero vi.resetModules() in apps/web/lib.