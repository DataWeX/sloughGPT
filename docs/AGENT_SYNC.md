# Agent Sync — read this first, update it when you push

**Purpose** — one place every session reads before starting: what has landed,
what is in flight, and how to sync your own work. Keep it short; update it in
the same commit that pushes your change.

**LANDED 2026-10-17** — card `bdfcb8cb` (single-source scene build,
branch `feat/scene-build` stacked on `feat/opcode-split`): the world
frame is no longer rebuild-bound. `domain/shell/_internal/scene_build.c`
— voxel mesh extraction + BVH construction — goes through the same
compile-once loader as trace_core.c (gcc `-O3 -fopenmp` → ctypes, NVRTC
→ RawKernel; `SLO_Q` the only target token). The mesher is one thread
per voxel; the BVH is an explicit-stack iterative build (NVRTC forbids
recursion) that appends nodes in exactly the python recursion's
pre-order. **Parity is exact by construction, not statistical**: the C
reproduces each double→float32 rounding once, and its merge sort's
(centroid, arrival position) total order equals `argsort(kind="stable")`
— `cycles.py` switched to stable in the same commit. Seam is minimal:
`BVH._build` replaces only `_build_recursive`; `_build_voxel_mesh` tries
native first; both fall back to the untouched python oracle on
`SLO_CYCLES_NATIVE=0` (dynamic) or a missing compiler. The frame path
uses the CPU engine (the scene is host-resident — a GPU build would
spend its win on PCIe); the GPU entries exist, compile, and are run
bit-exact vs python by tests. Measured (24x12x24 world, pop 8, pools 3,
160x120@16spp): build_scene **0.614s/57.2% → 0.064s/16.9%** (9.6x, no
longer the frame's majority), world frame **1.073s → 0.377s** — the
world frame now converges with the device frame (~0.39s);
`benchmark_world_render.py --frames 2` PASS (device 0.12s / world
0.25s). New `test_scene_build_native.py` (15 tests: mesh/BVH/hit parity,
fallbacks, NVRTC bit-exact); 238-test cycles+render surface green.

**LANDED 2026-10-17** — card `4de18e24` (single-source trace core):
the two-implementations liability is retired — there is no numpy walk *and* a
CUDA walk to keep in sync any more. `domain/shell/_internal/trace_core.c`
owns the ENTIRE path (BVH walk + GGX BSDF + emission + lobe sample + Russian
roulette) and is compiled twice by `domain/slolib/_internal/native.py`:
CPU `gcc -O3 -fopenmp` → ctypes (`entry_trace_cpu` = OpenMP parallel-for,
GIL released), GPU NVRTC via `cupy.RawModule` (`trace_paths_gpu` = one
thread per path). **`SLO_Q` is the only target-specific token** (empty on
cpu, `__device__` under `__CUDACC__`), so parity is structural, not tested
hope. Counter-based RNG keyed `(seed, path, depth, slot)` gives determinism
+ CPU/GPU parity without `libcurand` (still absent in this env). One thread
per path on BOTH engines = zero host dispatches per bounce and zero
alive-ray compaction, which is what finally kills the ~60us/python-call
dispatch floor that made the operator-by-operator CuPy walk slower than CPU.
Entries take a FLAT arg list (a struct-of-pointers does not marshal
reliably across ctypes and NVRTC); `domain/shell/_internal/_native.py` is
the marshalling seam — packs per-mesh BVH tables into global arrays
(`moff_node`/`moff_face` offset node indices, prim face-ids re-based
local→global, uniform K padded with -1, f32 plane), dispatches by engine,
returns `None` when nothing compiled so `render()` falls through to the
numpy oracle (`_trace_batch` kept intact as oracle + fallback + neural
path). `SLO_CYCLES_NATIVE=0` (escape hatch) is read **dynamically per
call**, never at import — an import-time constant bakes the flag in and
leaks onto the oracle path for every later suite in the process.
Measured (`benchmark_world_render.py --frames 2`, PASS all tiers,
160x120@16spp): native GPU **0.39s device / 0.81s world**, native CPU
0.42s / 1.37s (the two engines within **8%** — the near-identical target),
oracle 6.01s / 10.75s ⇒ **15x device, 13x world** vs the previous best,
~7400x vs the scalar baseline. Agreement: cpu-vs-gpu 5e-09 (float32 last
ulp; gcc fuses FMA, NVRTC does not); vs the f64 oracle it is statistical
not bit-exact — a f32-vs-f64 nudge flips which high-variance RR paths
survive, so divergence is confined to bright/metallic pixels (corr 0.25,
~0 on dark pixels). New `test_cycles_native.py` (11 tests); the 476-test
cycles+render surface is green. Next measured victim: the world frame is
now **rebuild-bound** (trace ~0.4s of a 0.8–1.4s frame) → the voxel
mesher + BVH builder goes through the same loader.

**LANDED 2026-10-16** — card `e4f15b5a` (cycles renders on the GPU):
`cycles.py` is input-driven across the device boundary — hot helpers
sniff their module (`_xp(a)`), host arrays in → host path out, device
arrays in → device path out; only `render()` crosses (one upload, one
download). `resolve_array_module(device|SLO_CYCLES_DEVICE|"auto")`
probes the accelerator's `array_module()` with a real kernel launch.
Two structural decisions made the GPU actually faster than CPU (the
first CuPy translation was 2.6x SLOWER): (1) **all samples fuse into
one trace** — `render()` stacks S×H×W rays into a single `_trace_batch`
call, amortizing the walk's per-call host syncs across every sample
(fused 2.95s vs 34.6s per-sample at 160x120x16); (2) **the fused
Möller–Trumbore RawKernel** — every Python-level cupy call pays a
~60us dispatch floor (measured), so the ~75-call operator translation
burned ~4.5ms per leaf iteration; `_mt_cuda` compiles the identical
math (same EPSILON rules, same 1e30 sentinel, same float32 cast) into
one launch, and the walk dropped its redundant compactions (leaf subset
via one nonzero; the dual-child push needs no subset — leaves' -1
children self-exclude). Merging the leaf/inner prim block wholesale was
measured and REVERTED: wasted inner-row pairs quadrupled CPU walk time.
Also fixed: `generate_rays` broadcast int64 origins from int-typed
`Camera.origin`, silently C-truncating every bounce position.
Equivalence: CUDA walk **bit-identical** to numpy walk (t + face) and
scalar-walk agreement ≤1e-6 (new `test_cycles_gpu.py`, 15 tests).
Measured (`benchmark_world_render.py --frames 2`, PASS both tiers):
device frame GPU 3.19s vs CPU 18.91s = **5.9x**; world frame GPU 7.05s
vs CPU 9.67s (warm 4.15 vs 6.18 = 1.49x); vs the scalar baseline
901x. Stream end-to-end at 160x120@16spp: 0.12 vs 0.07 fps (1.8x,
incl. VM tick + PNG); below ~50K rays the CPU still wins — the walk's
dispatch floor is ray-count-independent — so the tiny demo config stays
on `SLO_CYCLES_DEVICE=cpu`. Known env quirk: `libcurand.so` is absent
(harmless — renderer RNG is host-side by design). Next: batch-size-aware
device auto-selection, and a full traversal RawKernel as the follow-up
if the walk floor ever needs to drop further.

**LANDED 2026-10-16** — card `a3b39114` (StreamDevice + asm streaming app):
the world can be watched live **from machine code**. `world_stream.asm`
loops `step-sim -> render -> encode_png -> stream write`; Python is kernel +
drivers + socket only. New pieces: `StreamDevice(frame_fn)` in `vm.py`
(ConsoleDevice callback precedent — newest-wins ring, overrun counting,
`read()` pull, `write(data, tag)` push; the future web socket attaches at
`frame_fn`), `WorldDevice` in `vm_devices.py` (steps the real `Simulation`,
renders via `RenderBridge` into a shared `TensorArena`, `stats`/`info`),
the asm app `domain/shell/_internal/world_stream.asm`, and the runner
`python3 -m domain.shell._internal.world_stream --frames N --out DIR`.
Register file never holds an ndarray: world render puts a slot, `cycles
encode_png` resolves the foreign slot, stream receives bytes tagged with
the tick (Card A composability exercised end-to-end). Zero-frame runs are
guarded in the kernel — `LOOP` decrements before it tests, so a zero
counter would run away. Gates: new `test_world_stream.py` (18 tests)
+ 10 adjacent suites green except 5 stash-verified pre-existing failures
(4× `TestNPUVMDevice`; `test_kernel::test_register_open_close` —
`DeviceManager.open` returns int, test expects `.fd` handle, fails with
this work stashed too). Measured: 3.98 fps at 64x48@2spp, 0.18 fps at
128x96@4spp CPU; render cost stays gated by `benchmark_world_render.py`
(Card B). ruff clean. Next: GPU tier for cycles (`get_accelerator()`,
CuPy installed) to make the stream genuinely real-time.
**PUSHED** `origin/feat/vm-bus-handles` (all three cards A/B/C) —
re-merged with main (`d3df8179c`) and re-gated post-merge: identical
5 stash-verified pre-existing red, ruff clean, benchmark PASS (155×).
Main landing deferred: `main` is checked out at
`~/Documents/sloughgpt-api-std` with active uncommitted edits (another
session) — sync-recipe takeover needs a clean idle worktree; merge sha
to follow. Also flagged: `origin/main` and local `main` have diverged
(336 vs 164 commits) — repo-wide sync issue, not resolved here.

**LANDED 2026-10-16** — card `ce0ec0b7` (vectorize `cycles._trace_batch`):
every per-pixel Python loop in the tracer is gone. `BVH.intersect_batch` is
a frontier walk over all live rays (padded prim tables, precomputed edge
tables, explicit-cross Möller–Trumbore, exact `tnear <= t_hit` cull);
emission gathers per-material once; the BSDF bounce runs per-material groups
with a fixed RNG draw count (deterministic per seed inside the new sampler —
pixel-exact parity with the old scalar lobe-pairing is not expected);
`_intersect_scene` batches per-mesh with grouped barycentric/normal gathers;
`_refract` takes per-ray eta (columnarize eta AND cos_i — mixing (k,1) with
(k,) broadcasts to (k,k)). Scalar paths (`intersect`, `_trace_single`) kept —
`render_state_tensors` still runs them (follow-up candidate on the card).
Measured `scripts/benchmark_world_render.py` PASS: Gateway-default device
frame 22.8s, RenderBridge world frame 16.6s = **126×** vs the scalar
baseline (2875s/frame extrapolated from the measured 140–320s at
128×96@2spp); profile says the walk is now numpy's per-pop floor
(~15 pops/ray × ~0.5µs), so real-time is the GPU card's job
(`get_accelerator()`, CuPy installed). Gates: 275 tests across the 6
cycles/render suites green + repl/vm/world suites green except the
stash-verified pre-existing 4× `TestNPUVMDevice` and router PPM failures;
ruff clean. Next up: card `a3b39114` (StreamDevice + asm streaming app).

**LANDED 2026-10-16** — card `1c64882c` (VM bus-value discipline +
`encode_png`): the register file now holds machine-representable values
only, the FileDevice fd precedent extended to arrays. `CyclesDevice.render`
returns a **TensorArena slot id** (`domain/shell/_internal/tensor_arena.py`)
instead of a live ndarray; `resolve()` at each driver boundary accepts
legacy live arrays, so old Python callers pass through untouched. New ops:
`encode_png` (PNG bytes for a slot — composes with `FileDevice.write` from
assembly) and `free_slot`. `MultimodalDevice` takes the shared arena, so
**assembly can now do `cycles render → multimodal embed` through a slot**
(flagship test: pure-asm `render → encode_png → FileDevice.write` writes a
valid PNG). All bus/repl call sites updated to the slot contract. Gates:
new `test_vm_bus_handles.py` (36 tests) + 9 touched/adjacent suites green
except 4 pre-existing `TestNPUVMDevice` failures (device calls
`load_file`, test fake implements `load_model` — fails with this work
stashed too; another lane's drift). Next up: card `ce0ec0b7` (vectorize
`cycles._trace_batch` per-pixel loops), then `a3b39114` (StreamDevice +
asm streaming app).

**LANDED 2026-10-16** — card `7cefbbcf` (World–Model Bridge Stage A): babies
can be born with a `ModelBrain` (`domain/shell/_internal/model_brain.py`) —
interface-identical to `Perceptron` (forward/update/`W`/`b`/`H`/`bh`/
to_dict/from_dict), same RNG draw order (H, bh, W) and genome tensor
inventory, but the hidden projection is **trained by backprop of the in-world
reward** instead of frozen random. Opt-in: `WorldParams.brain_kind="model"` /
`world_driver --brain model`; default `perceptron` leaves every locked proof
bit-identical. Evolution/serialization/`decide_*` needed zero changes — the
seam is the duck-typed brain contract. Gates: 22 new tests + 290
evolution/simulation/world tests green post-merge; `scripts/benchmark_world_brain.py`
PASS (throughput model 55.9 vs perceptron 53.3 ticks/s = -5.0%, gate <20%;
emergence evolved 341.3 > frozen 216.6; in-life H drift 0.000168 vs 0.000000
control); ruff clean. Known unrelated red on main:
`test_world_render_router::test_image_returns_ppm` asserts PPM while the
endpoint serves PNG — fails with this work stashed too (pre-existing).

**2026-10-11 ~06:30 — PROVIDER ROUTER CHAT DRIFT FIXED (branch
`fix/provider-processor-flakes`, card 9e0483b3 item 4).** The lone remaining
failure the cleanup scan flagged as "needs owner": `test_chat_delegates_to_stream`
was RED because `ProviderRouter.chat` had drifted to `await text_provider.chat(...)`
(a duplicated processor loop with **no tool rounds**), while the Jul-25 test pins
the original contract — chat is a blocking adapter over `chat_stream`. Restored
the consumer form (all providers implement both protocol methods; prod has zero
router.chat callers — the HTTP layer uses `get_provider()` raw). Gates: target
1/1, test_provider_processors 49/49, Router-touching suites 567/567, adjacent
raw-provider suites 78/78, ruff check clean (file's `ruff format` dirt is
pre-existing at HEAD). slonet-path test already green at HEAD.
**PORTED 2026-10-14** — the fix was stranded on `fix/provider-processor-flakes`;
cherry-picked to main as `35cca61bd` under merge `af14b2231`. Re-verified on
merged main: both named tests 2/2, `test_provider_processors` 0F, 9
router-touching suites 917 green, ruff check clean; `ruff format` dirt
re-confirmed pre-existing against main's own copy.

**PORTED 2026-10-15** — three fixes stranded on the tokenizer lane
(`a1b3c2a92`, plus the write-behind follow-up `6abe7ada1`) that never reached
main, so all three defects were still **live on main**: (1) RAG per-document
O(corpus) index rebuild (`ProductionRAG.add_document` called
`retriever.build_index()` every insert — card `9327d5ba`); (2)
`knowledge_store` drop-before-insert data loss + O(N²) persistence
(`_save_entries`/`_save_visited` dropped the collection then re-inserted
every doc per add — a crash in that window left the store empty — card
`d35a8000`); (3) `quant_core` import-time crash when the `.c` source or gcc
was missing (unguarded `_LIB` deref turned a numpy-fallback situation into an
`AttributeError` at import — card `9c548bf0`). Hand-ported to main's layout
(`quant_core` lives at `domain/infrastructure/_internal/`, tests at
`packages/core-py/tests/`); main's `BM25Indexer.index()` keeps its own
reset-then-rebuild structure rather than the lane's local-variables rewrite.
Landed `4f28e31d2` under merge `f66e92893`. Measured: 400 single-entry
persists 0.393s → **0.018s**, per-insert cost flat vs corpus (was growing
0.31→0.98ms). Re-verified on merged main: 431 green across `test_rag` 52,
`test_vector_store` 72, `test_quant_core` 18, `test_knowledge_memory` 43,
`test_knowledge_ops` 67, `test_memory_service` 65, `test_learner_continual`
42, 6 memory/rag suites 72; `ruff check` clean on all changed files (the
file's `ruff format` dirt at `rag.py:381` is pre-existing on main). Two lane
test patches ported with the source: `_save_entries` → `_write_entries_json`
monkeypatch targets in `test_knowledge_ops.py`/`test_knowledge_memory.py`
(the count must track the actual store rewrite, not helper naming), plus the
`flush_json()` call in the `clear_all` test now that single-entry writes are
write-behind.

**LANDED 2026-10-15** — card `dd670ece` (high): `scripts/python` put only the
repo **root** on `PYTHONPATH`, never `packages/`, so `import downcraft` fell
through to the conda env's pip-installed **Sep-18 snapshot** (pip 0.2.0,
`site-packages/downcraft`). That copy's "already downloaded" branch is
unconditional, while the tree copy (`6c626f043`, an ancestor of main)
re-downloads when a "complete" state entry has no file on disk — so every
`$PY -m downcraft` consumer (`e2e_gateway_downcraft.sh`, benchmarks, anything
using the launcher) silently ran stale code, reported success for a download
that wrote **no file**, and the already-merged reget fix was dead in practice.
Only a `PYTHONPATH` entry beats site-packages, so the launcher now prepends
`$ROOT/packages/downcraft` — deliberately narrow, because only `downcraft`
actually has an installed copy (verified: no other `packages/` name is
shadowed), and replicating the whole `pytest.ini` set would drag pytest-only
entries (`.pytest-libs`, `tests/api-server`) into every invocation.
`tests/core-py/test_launcher_repo_copy.py` **ratchets** the list: it scans
`packages/` for importable names that also exist in the *resolved*
interpreter's site-packages and fails until each is pinned by the launcher, so
a future installed copy cannot silently shadow the tree again. Landed
`bd58cba40` under merge `2d427428b`. Gates: guard test **4 passed** (with the
launcher stashed it is 1 failed / 3 passed — the ratchet fires as designed);
broad import smoke **5/5** through the modified launcher
(`domain.shared/infrastructure/cognition/memory` + `downcraft`, the latter
resolving to the tree copy); `downcraft` contract test **6 passed**; `ruff`
clean. `e2e_gateway_downcraft.sh` fails at section 1 ("identity body !=
source") **identically with and without the fix** (baseline-verified by
stashing) — a pre-existing gateway-content mismatch, not this change; the
checkpoint fixture it needs (`checkpoints/step_25.pt`) is absent from this
checkout and is gitignored.

**LANDED 2026-10-15** — card `1f5dec62`: peer manifest serving. Server-to-server
model sharing was **half a feature** — `ExternalDownloadBackend` fetches
`{base_url}/models/{id}/manifest.json` and bails with `{'status': 'error'}` on
404, then downloads each file from `/models/{id}/file/{path}`, but this server
exposed *neither* shape (only `/models/file/{id}/{path}`). The client seam
worked against a stub peer; a real peer running our codebase could not serve.
Landed two routes plus a domain helper: `build_model_manifest()` derives
`{files:[{path,size,sha256}]}` from the same cache `resolve_cached_path` reads,
confirming each listed path is one the file route actually serves (the
resolver's 1000-byte floor means a sub-1KB file must **not** be advertised),
and reads checksums from the `.sha256` sidecars. Three things worth remembering:
(1) the manifest route returns the **bare** `{'files': [...]}`, not
`success_response()` — the peer does `json.loads(resp.read())` and reads
`manifest["files"]` at the top level, so an envelope makes it see zero files
and fail with "Manifest contains no files"; (2) both routes are registered
**last** so their greedy `{model_id:path}` prefixes cannot swallow
`/models/download/...` or `/models/file/...`; (3) the nested file route resolves
straight from the cache instead of delegating to `serve_model_file`, which calls
`get_backend()` bare — **pre-existing breakage**, verified failing on clean main
against `get_backend(name: str)`, left untouched here. A plain `FileResponse` is
a valid answer because `CompressedDownloader` passes bytes through when no lz4
`Content-Encoding` is advertised. Landed `078f8f01a` under merge `9d2106188`.
Gates: `tests/server/test_models_router.py` **50 passed** (6 new peer-contract
tests), `TestPrecision::test_cpu_path_uses_suggestion` fails **identically on
clean main** (pre-existing, unrelated), `ruff` clean; standalone e2e seeded a
snapshot cache and drove both routes → `E2E_PEER_MANIFEST_OK`.

**LANDED 2026-10-16** — cards `aecf2bb1` + follow-through on `dd670ece`: the
launcher now makes **every** `packages/` project resolve from this checkout,
not just `downcraft`. Scoping the card exposed a *second* competing resolver
the original fix never addressed: the conda env's `sloughgpt-0.3.0` editable
install registers an `_EditableFinder` whose `MAPPING` hardcodes the **main
tree** checkout, so from any worktree `import mogdb` resolved to the main
tree's copy rather than the one being edited — a worktree session would test
its changes against the wrong code, silently. The card's stated premise did
**not** hold, though: `mogdb` is not an installed snapshot (it resolves via the
editable finder), and `planner`/`strui`/`avion`/`chargectl`/`aml`/`infra_lib`/
`sloughgpt_sdk` are not installed at all — only `downcraft` ever had a real
site-packages twin. So the right fix was not "pin three names" but "the working
tree wins for every project", which covers both resolvers at once. Two details
worth keeping: (1) never a bare `$ROOT/packages` — that is a PEP 420 namespace
portion and loses to an installed regular package, a silent no-op that reads
like a fix; (2) the first loop used `$(basename …)`, forking per project and
costing **42ms on a 47ms baseline** — measured, then rewritten as pure-shell
parameter expansion (46ms). The ratchet test is now **behavioural** rather than
a regex over shell source: it runs `scripts/python` for every importable
`packages/` project and asserts the module origin sits inside this checkout's
`packages/` tree, which catches the editable-install redirect a string match
never could. Landed `743da3851` under merge `f562ecd2a`. Gates: 4 passed;
`mogdb` + `downcraft` both resolve to api-std (`mogdb` was the main tree
before); `routers.models` and `domain.infrastructure` import cleanly with 12
project roots injected; the downcraft reget guard still live.

**Last update**: 2026-10-09 — card `ff9ef505` → review: **generic embedding
provider config restored on main** — `api_key`/`base_url` + the
`openai_api_key` alias sync on `EmbeddingConfig`, `OpenAIEmbedder`
`base_url` wiring, and `tests/test_embedding_config.py` (8 tests).
Root cause: the feature landed on lanes as `bab6c42ed` (card `90839119`)
but main never received it, and `81616a6f7`'s sweep removed the same
fields from main — **41 local branches carry the tests but not the
fields (6 red each, incl. the root full-suite checkout)**.
Cherry-picked `bab6c42ed` onto main. Green: embedding 8, config 34,
consumer surface (vector_store/infrastructure/rag) 61+1 skip,
vector_router 37, ruff clean. Red branches heal at their next merge
from main.

**Last update**: 2026-10-07 — card `d994d4f8` → done: `GET
/knowledge/rag/documents` now pages (**default 200, `?limit`/`?offset`,
`?all=1` escape hatch**; envelope gains `total`/`limit`/`offset`;
`RAGService.list_documents` slices *before* building per-doc dicts) —
first page **31 KB vs 3,080,544 B encoded (99×)** measured on the live
19,703-doc index; the knowledge page Sources panel shows "Showing X of
Y" + Load more instead of downloading/rendering everything. Contract
baseline regenerated (+3 query params, +422; 54/556/492 unchanged).
Tests: 233 Python + 29 vitest + tsc/ruff green. Live: cherry-picked onto the root checkout's branch (now
`feat/dev-ai-unified`) as `57098d898` — 7 files; baseline + paging
tests stay on main — active from the next server restart.

**Last update**: 2026-10-06 ~10:30 — card `625d85e5` → done, **live gap
closed** (fix cherry-picked onto the root checkout as `98220d30f`;
live `/health` 10,099,639 B → 1624 B)
(investigation only, no new code; primary fix was already on main as
`2c460cbaf` since 10-01): systemic ≥1s slow-log tail **root-caused** —
`GET /health` shipped a ~10 MB `quantization.per_tensor` blob that
FastAPI JSON-encodes **on the single event loop** (2,106 calls/123 min
≈ 179 MB/min from 3–4 persistent pollers), stalling every in-flight
request: 61% of the 30s `SLO_DUMP_STACKS` stack dumps landing inside
stall windows caught the main thread mid-encode (baseline 21%), 25
same-second multi-endpoint slow clusters, 188/207 warnings were small
collateral victims. Amplifier: host CPU starvation from parallel
session test runs (PSI cpu peaks 75–80%); secondary: HF-hub
`getaddrinfo`/`list_hf_models` on request paths (`/datasets` 1.83s for
36 KB). **LIVE GAP CLOSED 10-06 ~10:08** (user-approved): the fix's
health files were cherry-picked onto the root checkout's
`feat/soul-path-grammar` as `98220d30f` (patch applied clean despite
that branch's Field-registry divergence; 49/49 + 3/3 tests on-tree;
radar-polygon half left for the regular merge) — live `/health` after
restart = **1624 B** with `model_loaded=True`, no `per_tensor`.
Same window also produced COORD card `c550503c` (live :8000 wedge
cycle evidence: accept queue 0→191, faulthandler dumps starved
mid-write = GIL-held block; OOM/journald ruled out). Follow-ups
filed: `d994d4f8` (rag/documents 2.8 MB paging), `14d26c75` (HF-DNS
cache/fast-fail), `8db4031b` (slow-log carries PSI/loadavg). Method +
evidence in the card (journalctl + faulthandler dumps; py-spy blocked
by ptrace_scope=1).

Earlier ~03:05: `feat/board-read-column-guard`
**lands on main** (fast-forward, card `b538ecbd` → done): `load_board()`
now validates every card against the board's **own schema-header**
vocabulary — an undeclared column (hand-typed; the file is editable by
design) logs a warning and re-homes under `todo`, instead of letting
renderers silently drop the card (`board show` iterates header columns
only; stats still counted it → the two views disagreed). Never raises:
one typo must not break the board. Completes the split fix: write-path
`validate_column()` = `67c406ee2` (card `46f0fac4`, other session), data
repair of the 3 `in-progress` orphans = `86fb9a300` (same session);
read-path guard was the missing half. Note the branch vocab divergence
(main header `todo/wip/review/done`, root header `.../in_progress/...`)
— validating against each board's own header makes the guard
branch-agnostic; both live boards are internally consistent today (0
coercions). Verified: 5 new tests (`test_board_column_guard.py`,
incl. a repo-board canary) + full app-planner suite **205/205**.

Earlier ~02:10: `feat/stats-local-day` **lands on
main** (fast-forward, card `7e708a68` → done): the stats "today" count
no longer reports 0 during the first hour after local midnight.
`Note.date_str` sliced the **UTC**-stamped `created_at` while
stats/filters compare against `date.today()` (**local**) — during
00:00–01:00 WAT the two dates disagree (a midnight-only flake,
invisible to daytime CI; `test_stats_endpoint` failed only in that
window). Fixed at the single derivation point: aware stamps now convert
to the local calendar day (`astimezone().date()`), naive/malformed
stamps keep the historical slice, so all four comparison sites plus the
export/grouping displays inherit local-day semantics in one place. Ids
and stored data untouched; kanban `card.due_date` (a different date
source) deliberately not touched. Evidence: live before/after with
`TZ=America/Los_Angeles` forcing the mismatch — endpoint `assert 0 == 2`
FAIL → PASS; full app-planner suite **200/200 in default TZ and TZ=LA**;
6 new deterministic `TZ`-forced unit tests
(`tests/test_note_date_str.py`, both conversion directions + fallbacks).

Earlier ~01:25: `feat/kernels-nogil` **lands on
main** (fast-forward, card `f30be275` → done): all 16 builder kernels in
`domain/training/_internal/slonet_kernels.py` are now
`@njit(cache=True, nogil=True)` — a thread-domain call to any hot kernel
**releases the GIL** instead of starving the host loop for the whole run
(the demonstrated outage mechanism, now measured at real scale: a stock
prompt-scale `fused_attention_multi` call held the GIL for all ~42 s of
its duration). New gate: `scripts/probe_gil_hold.py` (the card's
referenced `benchmark_isolated_execution.py` never landed) answers the
decision question directly — standalone twins: control thread stalled
**100%** without nogil vs **0.5%** with it; `--real` mode probes the
production decorator: **0.4%** of a 2.6 s call. Numerics are
**bit-identical** (6/6 kernels, `max_abs=0` stock vs nogil);
`benchmark_slonet.py` **7/7** (`logit_diff 0.0000`, `max_output_diff
0.0000`); kernel suites 122 passed in both env protocols with the 2
failures proven pre-existing/by-design (`test_causal_mask_effect` fails
on the stock file under numba-hidden; `test_check_numba_false_branch`
asserts numba-absent). First run recompiles (cache-signature change, one
time, expected). Findings for follow-up: (a) numba is **user-site only** —
the repo-standard `PYTHONNOUSERSITE=1` protocol silently runs the numpy
fallback and never exercises the compiled kernels; (b) the
`benchmark_latency.py` gate could not run — the shared `:8000` `/chat`
hangs while `/health` reports healthy (reproduces after a server restart;
PGQ-wedge territory, another session's lane) and the host sat at 97°C
thermal clamp; (c) pre-existing perf oddity: prompt-scale
`fused_attention_multi` (512×512×8×64) takes ~40 s/call **stock and
nogil alike** — the docstring's einsum note understates it at that shape.

Earlier ~12:55: `fix/vector-shm-tracker` **lands on
main** (fast-forward, card `ccad389e` → done): VectorBE's `/psm_*`
resource-tracker warning spam is fixed at the **topology** level —
`VectorBE.__init__` now calls `resource_tracker.getfd()` before starting the
pool, giving the fork arm the same shared-daemon wiring forkserver
(`forkserver.py:92`) and spawn (`spawn.py:119`) children already get.
Root cause: `SharedMemory` registers *attaches* as well as creates
(gh-82344); fork workers forked before the parent first touched the tracker
inherited `_fd=None` → a private tracker daemon per worker → exit cleanup
re-unlinked every attached segment (118 warning lines from one
matmul+softmax+rmsnorm probe). Sharing the creator's daemon makes attach
REGISTERs idempotent against the creator's claims, so the create/unlink
pairing alone governs cleanup. An earlier attach-site `unregister()` fix was
proven *wrong* by topology evidence — in the shared topology it clobbers the
creator's claims (8 KeyErrors, one per segment). Verified: **0
`resource_tracker` lines in fork / forkserver / spawn arms**, no `/dev/shm`
leaks, `test_vector_backend.py` **24/24 in default and fork-env runs**; the
suite's 2 env-hatch blind spots fixed (defaults test + `vector_be` fixture
now `delenv` `SLO_VECTOR_START_METHOD` — the fixture pin also guards the
documented live-fork-in-suite deadlock). Follow-up on the card:
`scripts/benchmark_vector_backend_pool.py` (the finder-session script the
card references) still does not exist.

Earlier ~08:15: `fix/journey-server-skip` **lands on
main** (fast-forward, card `b9f12ad2` → done): the serverless gate block in
`packages/core-py/tests/test_user_journeys.py` +
`test_computer_use_training_integration.py` is now **124 skipped in 12s,
EXIT=0** (was ~113 errors plus a 90s × 2 dead wait) — both
`ensure_servers_ready()` fixtures now do the repo-blessed capability gate (5s
probe → `pytest.skip("API server not running")`, the `test_integration.py` /
card `6e826226` convention) instead of asserting live servers. Chromium itself
was already healthy (installed 10-02; live launch `151.0.7922.34` — the card's
original "missing executable" premise was stale). SECOND defect found while
verifying: `test_user_journeys.py` was **uncollectable under
`--import-mode=importlib`** (root-ini addopts) — importlib's rootdir-relative
`tests.*` module name resolves through cwd's repo-root `tests/` **package**
(it has `__init__.py` + its own conftest), so `packages/core-py/tests/conftest.py`
never executes its avion `sys.path` insert → `ModuleNotFoundError: avion` at
collection (probe evidence: avion never in sys.path under importlib, present
from sessionstart under plain mode). Fixed file-locally with the same
source-bootstrap idiom both conftests already use (no installs, mode/cwd
independent); the underlying duplicate-`tests/` naming collision remains card
`d2803a3c`'s scope. Verified: both modes 124 skipped EXIT=0; ruff +
py_compile clean.

Earlier ~03:27: `fix/root-tests-order-dependence` **lands
on main** (fast-forward `0246b027c`, card `5473eb90` → done): the rotating
order-dependent `tests/` failure is **root-caused — it was never a
module-level global; it was a path-depth off-by-one**. `tests/test_api_keys.py`
computed `_server_dir` with `parents[2]` (the idiom for `tests/server/`-depth
files), which at `tests/` depth overshoots to the PARENT of the repo — in a
nested worktree (the gate layout) that's another checkout, so it put the parent
repo's `apps/api/server` at `sys.path[0]` **twice** (the second insert was
unconditional). `training` then resolved cross-tree, `find_repo_root` read the
parent's `data/`, the dataset lookup missed, and `raise_error(400)` escaped the
tests' bare `FastAPI()` as an unhandled **500** — so `>=400` assertions passed
while `==200` ones failed: the rotating signature. Plain checkouts get a dead
path (dormant → "stopped reproducing"); `tests/server/` scope was accidentally
protective (imports `training` first, `sys.modules` shields the rest). Fix:
`parents[1]` + drop the duplicate insert; same class in
`packages/core-py/tests/conftest.py` (`parents[2]` → `parents[3]` — dead
everywhere, while line 23 already had `parents[3]` right for avion); both
`_make_client` builders now use conftest's `build_test_app` so `raise_error`
yields 4xx, not 500 (the `ddd3b3d49` precedent). Gates: deterministic **2-file**
repro 3 failed → 33 passed; non-server scope **1224 green** (was 1221+3); full
scope **3166 green**; plain-layout worktree 1224 green; core-py collection exit
0; ruff clean. `chat_trainer::test_generate` never reproduced after `ddd3b3d49`
(green in every full run since) — no mechanism was ever observed for it, so the
card records it as not-reproducing, not fixed.

Earlier ~19:20: `fix/core-py-feedback-suite` **lands on
main** (fast-forward `ce02af883`, card `37325860` → done): the 8 pre-existing
feedback-suite failures are fixed, and **both clusters were test-vs-impl
drift, not code bugs** — no production code changed. Cluster 1:
`meta_weights.py:8` / `model_health.py:14` carry the `from __future__ import
annotations` ratchet, so dataclasses record `f.type` as the *string*
`"float"` and `assert f.type is float` could never hold; the tests now
resolve with `typing.get_type_hints()` (repo-wide grep: no other file uses
the f.type-identity pattern). Cluster 2: `k` defaults to **5** — schema,
web controller (`k = 5`) and `meta-weights-controller.test.ts` all agree, the
test's `10` was stale — and the router registers `/meta-weights/ping|get|stats`,
exactly what `docs/routers.md` documents. Gates: the 2 files **153 passed**,
ruff clean. Two findings for other cards: (a) the 3 health failures that
surface next to them are pre-existing on pristine main *and* order-dependent
(8 when run alone → 3 with siblings) → card `5473eb90`; (b) core-py's
documented `-n auto` gate (`docs/TESTING.md:10`) cannot run here —
**pytest-xdist is not installed** (install needs user approval), and the
suite is 42,568 tests, so serial runs exceed 30 min.
Earlier ~16:45: `fix/board-column-validation` **lands on
main** (fast-forward `67c406ee2`, card `46f0fac4` → done): the column name is
validated at the **core**, not at the CLI. `PlannerStore.validate_column()` is
now the single source of truth and every write path funnels through it —
`add_card`, `move_card`, and `update_card(column=)` — so a retired or
misspelled spelling (`in_progress`, `in-progress`, `TODO`) can no longer enter
the board; that gap is exactly how stale spellings had leaked in (only
`_board_move` ever checked). `column_names()` stops at the schema header with
a substring pre-filter, so the hot path keeps its throughput: **58µs** on a
1.23 MB / 1052-card board vs 36.1 ms for a full `load_board()` = **0.1% of an
add**. CLI `_board_add`/`_board_move` catch `ValueError` for the friendly
`Invalid column: X. Valid: …` message and the duplicated check in
`_board_move` is gone. Gates: app-planner **194** (+9), ruff clean; benchmark
no regression — faster than unmodified main in 3 of 4 paired runs while the
box sat at load 10–12 (the swing is contention, not the code). Board half on
root branch `86fb9a300`: 3 hyphenated `in-progress` orphans → `in_progress`
and the card → done, 957 cards / 959 lines preserved, 0 invalid columns left;
`verify_board` still reports only the pre-existing `todo:78` break
(2026-10-04T06:36Z, identical at HEAD — not introduced here, journal is
append-only so it is reported rather than rewritten). Blast radius checked:
`PlannerStore` has no caller outside app-planner, and the GUI runs on
`KanbanStore`, which already guards its own `move_card`.
Earlier ~10:15: `feat/wip-column-rename` **lands on
main** (fast-forward `3f8465164`, card `ffaea823` → done): the kanban column
is **`wip`, never `in_progress`** — `config` maps both directions, the default
column schemas in Python **and** TS, 4 test files, package README + planner
specs, and 31 board rows re-keyed (3 `in-progress` orphans folded in) with the
card hash chain re-sealed because `chain_hash_for()` hashes `column`.
Gates: app-planner **185**, web planner **89**, tsc/ruff clean, chain `[]`.
Hard cutover — no compat alias. Earlier ~10:05: `feat/pugqeep-forkserver` **lands on
main** (merge `d996f8ebc`; cards `df5affc7`+`e69915c1` → done, notes → done):
gate run #4 green 621F ≤ baseline 683F, playwright-class E 0, 0 timeouts, no
hang; post-merge targeted re-run = identical F-set, E 0 (see Landed). Earlier ~05:30: `feat/landing-7-cards` lands (7
infra-review cards + campaign `d083e732`, cherry-picks `2127abb6e..344168b4f`,
see Landed): startup finalizers + MogDB history + webhook facade + profiler
feed + test hygiene — tests/server **1942/1/0 vs main's 5-failed baseline**.
Earlier ~05:05: naming consolidated (card `43ca5222`):
**Mole is the one app** — never "doctor"/"watcher" as names; docs realigned
(TESTING/INDEX/FEATURES), legacy `/doctor` paths grandfathered until the
unified `mole` CLI. Earlier ~04:45: `feat/mole-watcher` lands (card `e0c80774`,
cherry-pick `e9fb80cec`, see Landed): Mole phase 1 — always-on, no-AI
monitoring (cadence, identity-set delta dedupe, JSONL journal, load-context).
Canonical 4-tree gate ≈ main baseline (833/839 nodeids
identical; the 6 extras pass standalone → shared-`/tmp` test-isolation defect,
card `6369c03e`). Earlier the same night: `feat/avion-transcript-batching`
landed (card `68ea8b9d`, cherry-pick `371025464`, see Landed): group-commit
transcript flush, benchmark green at baseline (quiet window, load 1.6); and
`fix/boot-overlay-stall` (card `cb089b43`, cherry-pick `8acb530be`, see
Landed) — the boot overlay can no longer hang forever; firefox ux-flows
benchmark **13/13**. Earlier 2026-10-03: `fix/journey-test-gates`
landed on user sign-off (merge `e081ea9f4` + test stabilization `175d7db8c`,
see Landed).
Earlier the same day: the **avion stack** — cards `13f50db7` journeys takeover
+ `e26dc68c` drivers + `24676ff2` event logger + `553da7a7` agent loop, see
Landed — and downcraft compression (card `19cd41dc`); journey suites repointed
to `:5173` (gate9's route-smoke F-class was a stale port, not a missing
server). Gates and benchmarks now run on conda `sloughgpt` + `PYTHONPATH` —
the root `.venv` is NOT authoritative (its green was hiding 2 zstd failures;
`zstandard` is now installed into conda, user-approved).

**2026-10-16 — startup benchmark gains the launcher path (branch
`feat/notebook-web-ui`, card `85e15cc4` follow-up).** The card's serve
launcher was landing in a concurrent session (`22cb8d1c4` + fixes
`2e09f1a70`, `dcc0165c0`, `e883379bf`) while this session independently
rebuilt it — duplicate caught at the gate by a path-based check
(`git log main -- <owned paths>`, not commit-message grep), this session's
implementation **discarded**, theirs kept. What survives is the measurement:
`scripts/benchmark_startup.py --via {uvicorn,serve_ui}` boots the API through
either the raw-uvicorn baseline or `scripts/serve_ui.py` and reports
`time_to_health_s` / `time_to_ready_s` / `stop_latency_s` (shared
`benchmark_results record` history). Same load window: health **3.01 s**
baseline vs **3.89 s** launcher, ready 123.0 vs 123.9 (the app's own warm-up
dominates both), stop **25.11 s → SIGKILL** (baseline — the app lifespan's
PGQ engine swallows SIGTERM) vs **2.23 s** (launcher: bounded TERM→KILL +
reap). Gate = `stop_latency_s ≤ 14 s` on the launcher path only, i.e. its own
contract (wrapper escalation 5 s, `terminate_grace` 10, `stop()` reap 12).
Landed by ff from `~/Documents/sloughgpt-api-std` (see Sync recipe). Deltas
deliberately **not** landed here — port/host overrides and reuse-ownership of
an already-running stack — are handed off in the card note for whichever
session owns `scripts/serve_ui.py`.

**2026-10-16 — MogDB full-sync mirror encode 2.6× faster (branch
`feat/mogdb-sync-perf`, card `223f8de1`, source card
`20260916_100007_mogdb-performance`).** `MogDB(sync_dir=...)` wraps every
collection in `SyncableCollection` with the default `sync_mode="full"` — i.e.
**every insert/update/delete rewrites the whole JSON mirror** — and the encode
was `json.dump(docs, f, indent=2)`: incremental Python-level streaming plus
pretty-printing, on the per-write hot path. Now one
`json.dumps(docs, separators=(",", ":"))` + a single `f.write` in both the
plain and gzip branches; semantics unchanged (temp file + atomic `replace`,
`json.load` contract — every reader is parse-based, no pretty-text consumer).
Measured interleaved A/B (3 pairs, stash-swapping only this file, normalized
to the benchmark's own `JSON write` machine reference because load swung 2–4×
inside the window): full-single single-write **47/51/56 → 197/159/137 ops/s —
ranges non-overlapping, 2.6× normalized**; encode micro 63.7 → 25.2 ms per
1000-doc mirror, file 226,625 → 164,624 B (−27%). Gated by 3 new deterministic
contract tests (compact single-line mirror; encode-once counted **through the
`json_sync` module binding only** — the collection's own journal writes go
through the shared `json`, so a global counter lies; one atomic replace per
write): `packages/mogdb` **283/283**, consumer `tests/test_api_keys.py`
green, `ruff check`/`format` clean on both files. NOT mine, pre-existing on
main: `test_slonet_provider_real.py` 2 failures (`ImportError: LayerNorm`) and
`collection.py` `ruff format` dirt.

## Sync recipe

```bash
git fetch origin
git merge origin/main            # in your worktree — keep yours current
# ... your work on feat/<name> ...
git push -u origin feat/<name>   # push your own branch when done
```

- `main` is checked out at `~/Documents/sloughgpt-api-std` — land feature
  branches to `main` from there (it was clean and idle when last used); say so
  on the kanban card first if you take it over.
- Shared toolchain: ONE root `.venv` (worktree `.venv` entries are symlinks to
  it), ONE root `node_modules`, ONE `packages/*` + `apps/*` tree. Never rebuild.
- After landing: update this file, move your kanban card
  (`python -m app_planner …` + `sync`), and keep `docs/INDEX.md` current for
  new docs.

## Landed on main (origin/main = `344168b4f`)

- **2026-10-04 · `feat/wip-column-rename` lands — the kanban column is `wip`,
  never `in_progress` (hard cutover).** Card `ffaea823` → done; ff `3f8465164`
  (`.wt-wip` worktree). The column name is a **primary key** joining card data,
  `config.STATUS_TO_COLUMN`/`COLUMN_TO_STATUS`, the default schemas in Python
  *and* TS (`store.py`, `kanban.py`, `apps/web` planner helpers/BuffetEngine/
  types) and every card's `chain_hash` payload — all moved in one pass; 4 test
  files, the package README and the planner specs follow. `migrate_boards.py`
  / `migrate_planner.py` keep `in_progress` only as a **legacy source key**
  mapping onto `wip` (a re-run must not resurrect it); `wip` beats `doing`
  everywhere (110 note entries use `wip`, **zero** use `doing`, and the web GUI
  already emitted `wip`). Data migrated byte-surgically — **952 card rows
  preserved**, 31 renames incl. the 3 `in-progress` orphans — then re-sealed
  with `compute_chains()` (the same call `sync()` makes by design) because
  `chain_hash_for()` hashes `column`: 29 → 0 violations, and pristine main was
  already 3-red (`[-1, 911, 912]`). Gates: app-planner **185 passed**, web
  planner **89 passed**, tsc clean, ruff clean, 0 eslint errors, `board stats`
  → `wip: 30`. **Gotchas:** (1) `.kanban/slot_history.jsonl` (root branch)
  keys slots as `column:pos` *inside* `node_hash()` — an append-only ledger,
  never rewrite its keys; (2) `board add` does **not** validate `--column`, so
  a stale session can reintroduce `in_progress`; (3) the long-lived
  `feat/create-router-projection` branch still carries `in_progress` and will
  conflict on `config.py` when it merges main.
- **2026-10-04 · `feat/pugqeep-forkserver` lands — fork→forkserver kills the
  fork-into-multithreaded class.** Cards `df5affc7` + `e69915c1` → done,
  notes → done; merge `d996f8ebc` (`.wt-l2` worktree):
  `SubprocessProcess` default `fork` → `forkserver` — the fresh child kills
  the fork-into-multithreaded-pytest deadlock class (gate run7: 60
  futex-stuck children). Module-level `_subprocess_worker` replaces the
  unpicklable `_worker` closure; pre-flight pickle check hard-fails
  (`PicklingError` before any allocation — closures/lambdas can't cross);
  `fork` stays as the explicit escape hatch; forkserver preload
  `[__main__, engine, target-module]` (steady spawn ~16 ms vs ~523 ms/child
  without — `scripts/benchmark_pugqeep_spawn.py`). Extended this session with
  card `a0ba949b`/`e69915c1`: **`VectorBE`'s bare `mp.Pool` (second fork site
  of the same class) → `mp.get_context("forkserver")` + `SLO_VECTOR_START_METHOD`
  escape** — it was the 88% gate wedge (isolated repro: parent `do_wait`,
  futex-dead workers; benchmark: one-time ~1.4 s cold start, steady state at
  parity — `scripts/benchmark_vector_backend_pool.py`); plus the
  `test_linux_cmds.py` `host`-fixture cwd leak fixed (bare `os.chdir` →
  `monkeypatch.chdir`; both gate runs sat in `test_time_no_args0` until exit —
  hygiene test `TestHostFixtureCwdHygiene` guards it) plus the same class in
  `test_shell_repl_more.py`: 3 bare `repl._cmd_cd("" / "~")` tests leaked
  `$HOME` from idx ~27600 (run #3 saw pytest cwd=`/home/mana` live) →
  `monkeypatch.chdir` + file-end `TestCwdHygiene`, and the shm
  `resource_tracker` warning spam carded as `ccad389e` (pre-existing, both
  start methods). **Gate run #3** (05:23→06:40, 77 min, no hang):
  597F/175E vs baseline 683F/0E — vector/linux_cmds/timeout classes all
  **0**; every one of the 175 E is the journeys family from
  `ModuleNotFoundError: playwright` (PYTHONNOUSERSITE hides user-site
  playwright; baseline chunks ran venv without -s) → run #4 carries
  `/tmp/opencode/pyshim` (playwright+greenlet+pyee symlinks on PYTHONPATH,
  no install). **Gate run #4** (07:05→08:34, 89 min, `gate exit=1`, no
  hang, **0 timeout blocks**): **621F/41902P/2E vs baseline 683F/0E** —
  vector/timeout/linux_cmds classes **0**; playwright-class **E 0**
  (`test_user_journeys` 103/103, `test_computer_use` 21/21; node-id diff
  removed 141 of run #3's E node-for-node). The 2 residual E = a
  pre-existing missing `route` fixture (identical ERRORs in run #3;
  `ROUTES`/`ALL_ROUTES` constants orphaned, zero `parametrize` in the
  file) → fixed post-gate with `@pytest.mark.parametrize`, E=0 verified
  by targeted re-run. The new `TestCwdHygiene` guard caught a 4th cd
  leaker (`test_pwd_after_cd` bare cd into `tmp_path`, leaving cwd in an
  empty dir → `test_tui_repl::test_dot` found no cwd dotfiles; run #3's
  cd-`$HOME` leak had masked it) → `monkeypatch.chdir` pin, both GREEN
  targeted; `test_phoneme_cli` 8F also disappeared (they had run against
  the leaked tmp cwd). `comprehensive_training_journeys` F = the deferred
  `:3000` hardcode class (card `e47e19ee` — `BASE = localhost:3000`,
  server died mid-file in run #4 after its first 9 tests passed; fully
  down post-gate, so its 19 parametrized routes join the same class) →
  post-gate tree ≤640F, still under baseline. Caution: the live root
  journal was externally reverted to HEAD mid-run (08:27, a session on
  `feat/create-router-projection` is still writing it) — run #3/#4
  evidence therefore lives in the WORKTREE journal only; `planner sync`
  is also non-idempotent (flips 2 foreign cards per call — do not run).
  Latency benchmark +414.9% vs baseline = contention artifact
  (load 19.6, 71 foreign pytest procs; live `:8000` serves root code, not
  this branch) — re-run when quiet. **Reconcile before
  merging**: this edits main's multiprocessing engine
  (`domain/infrastructure/_internal/pugqeep/engine.py`), while the root-repo
  session — now on `fix/startup-finalizers` at `ae237996e`, **225 commits
  ahead of main** — has already replaced that whole paradigm: `18c720be5`
  (owned `os.fork()` + framed socketpair channel, multiprocessing gone from
  their PGQ) + `d2454c84d` (single Pipe admission door) + `c4ecb6ff8`, all
  on `packages/core-py/domains/infrastructure/pugqeep/engine.py`. Root
  `AGENTS.md` on that branch now states the rewrite as fact ("Build from
  scratch, import last"). Different paths ⇒ textual merge is clean, but the
  two designs collide semantically on `SubprocessProcess.start`. This branch landed
  2026-10-04 with user sign-off — `SubprocessProcess.start` on main is the
  survivor; reconcile when their branch arrives. Benchmark names do NOT collide
  (theirs `scripts/benchmark_pipe.py`).

- **2026-10-04 · `fix/meta-weights-request-params` lands — requested sampling
  params are honoured again.** Card `d2dda007` → done, commit `806073e15`
  (onto `a189d78eb`). Contract chosen by the user: **feedback nudges only the
  parameters the caller left at their default; an explicitly-set parameter
  passes through verbatim.** Before this, `_apply_meta_weights` returned
  `get_adjustment()`'s absolute values, and `get_adjustment()` never receives
  the request — so with an empty feedback DB *every* request answered with
  0.7/0.85/40/1.15 whatever was asked, while `GenerateRequest` still validated
  the field (`ge=0.0, le=2.0`) and telemetry logged the request value the
  provider never received. Changes: `MetaWeightManager.neutral_weights`
  exposes the baseline so callers derive a *delta* (empty store ⇒ exactly zero
  change rather than a silent default swap); `_apply_meta_weights` gains
  `explicit` (pydantic `model_fields_set`; the WebSocket derives it from keys
  present in the frame) — **5 call sites, not the 4 the card claimed** (chat
  `:2971` was the fifth); the 5s cache now stores the **nudge**, never the
  merged result, because the nudge depends only on message+user while the merge
  depends on that request's explicit set (caching merged output would leak one
  request's explicit set into another's); the 4 `capture()` sites record
  `gen_params["temperature"]`. Tests: dropped the `_apply_meta_weights`
  passthrough patch (real code now passes explicit params through — strictly
  stronger), added nudge-applies-at-default and telemetry-records-sent-value,
  plus an autouse nudge-cache clear. **Benchmark** `scripts/benchmark_meta_weights.py`
  (new): cache HIT p50 1.64µs vs cache MISS p50 39.15µs (23.9×) — the merge
  cannot surface end-to-end. Gates: `tests/` 3156 passed / 0 failed,
  `apps/api/server/tests` 1223 / 0. **Found while verifying, proved NOT mine on
  pristine `a189d78eb` → card `37325860`:** the hygiene ratchet's
  `from __future__ import annotations` stringifies dataclass annotations, so
  `assert f.type is float` can never pass (3), plus 5 stale meta-weights
  *router* tests where `docs/routers.md:594-596` sides with the impl.

- **2026-10-04 · `fix/main-gate-green` lands — main's default test gate goes
  green (7 root-cause buckets).** Card `cbd2ffc3`, commit `ddd3b3d49` → landed
  as `70b162449` (rebased onto `81fd4d554`, moodboard-only delta, no overlap).
  Baseline on `c92adc47b`: **11 failed + 5 errors (16 red) in `tests/` → 3147
  passed / 13 skipped / 0 failed / 0 errors, three consecutive runs**;
  `apps/api/server/tests` **6 collection errors → 1223 passed** (the directory
  was dormant, off-`testpaths`, and never ran anywhere). Root causes, not
  counts: the contract gate was red because `routers/doctor.py` reached into
  `domain.core._internal.doctor{,.report}` (retargeted to the public facade;
  `default_report_path` joined the lazy map + `__all__`); `test_inference_generate`
  ×6 patched `domain.models._internal.provider.get_provider` while the router
  binds `get_provider` from `domain.models` **at import** — an inert mock — and
  the bare test app registered no exception handlers, so `raise_error()`
  propagated instead of returning 503; `test_cli_chat` ×5 imported `CLILogger`
  from `domain.logging._internal`, which never re-exports it; `test_feedback_domain`
  ×2 hit `_compute_gradients`' `engine is None → {}` guard (now driven by a stub
  engine, with a genuine sign invariant replacing a tautological assertion);
  `test_rag` ×1 was a **real product bug** — fusion applied `dense_weight=0.7`
  to a channel `ProductionRAG` disables by default, capping `combined_score` at
  0.3 so `HallucinationDetector.detect`'s `min_score=0.5` gate could never pass
  (grounding was structurally impossible, confidence always 0) → weights now
  renormalised over active channels; the `apps/api/server/tests` ×6 was a
  `tests` package-name collision → canonical `apps.api.server.tests` path
  (already used by `tests/server/test_server_api.py`). **Benchmarks:**
  `benchmark_bm25` A/B vs `c92adc47b` — recall@k 0.9833, MRR 1.0, reranked MRR
  1.0 **byte-identical**, latency within noise. Follow-ups filed: `b83a5788`
  (online LoRA is a silent no-op — the engine is never attached, yet stats
  count phantom updates) and `d2dda007` (meta-weights **replace** request
  sampling params — `get_adjustment` never receives them, so it cannot blend).
  **Gotcha:** the root repo's `app_planner` `.pth` has no `compute_chains`, so
  board writes made from the root copy leave that branch's board unchained
  (pre-existing there — base `f27654e54` predates the chain work); run board
  commands with `PYTHONPATH=<worktree>/packages/app-planner/src` from a
  chain-aware copy, and never issue two `board add` calls concurrently — the
  second silently clobbers the first.

- **2026-10-04 · `feat/landing-7-cards` lands — the infra-review follow-ups,
  surgical cherry-pick onto main.** Cards `223001e4` `d1f544fb` `ad9ef322`
  `b753cad5` `8349d901` `65d9e7ee` `832efdda` (+ campaign `d083e732`), 8
  commits `2127abb6e..344168b4f`: boot finalizers ⑨⑪⑫⑬ (history deadlock
  fixed; gzip at the seam — 4 MB stall 269.6 → 4.5 ms), startup history →
  MogDB (12 records migrated, JSON kept `.bak`), `startup_webhooks` →
  stateless facade over `WebhookStore` (shape-oracle pinned), profiler feed ⑩
  (per-hook timings, non-mutating `get_summary`, module-level `profile_hook`),
  test hygiene (stale patch targets at consumer read points + TestExecutor
  precondition fixture). Lineage fork `f27654e54`, so every pick was resolved
  against main's newer machinery: **main's REST contract, public module paths,
  and plain `WebhookStore.register()` win; only my `+` lines land** (one of my
  new tests adapted JSON-body → query-param POSTs). **Gates:** tests/server
  **1942 passed / 1 skipped / 0 failed vs main's 5-failed baseline** (+37 new);
  root `tests/` 13f/5e byte-identical to main's baseline (pre-existing); ruff
  clean; compression benchmark ~60×. Gotcha: `apps/api/server/tests` glob has 6
  order-dependent `tests.test_support` collection errors — pre-existing on
  main (A/B proven), off testpaths.

- **2026-10-04 · `feat/mole-watcher` lands — Mole phase 1: always-on, no-AI
  monitoring.** Card `e0c80774`, cherry-pick of `1b1ce40c6`: new
  `domain/core/_internal/mole/` — `run_watch` probes on a cadence over the
  existing `run_doctor`/`PROBES` (probe registry untouched; a newly
  registered probe is picked up next tick). Findings are fingerprinted by **identity set**
  `(source, check, severity, component)` — jittering payload counters (p95,
  health score, frame sizes) never re-alert (a content-hash re-alerted on every
  tick in the live smoke — caught before landing); every tick journaled to
  `$SLO_MOLE_JOURNAL` JSONL with `context` (loadavg + cpu_count — context,
  never a finding); events fire only on change; a failed tick is journaled and
  contained (the loop never dies mid-run). CLI
  `python -m domain.core._internal.mole` (`--interval/--max-ticks/--skip/
  --journal/--strict/--quiet/--json`); suggest-only, never applies. Docs:
  TESTING.md section + INDEX row (FEATURES deferred until it has an app
  surface). **Gates:** mole 6 + doctor 41 + router 9 green, ruff clean, CLI
  smoke (tick 1 baseline / tick 2 silent on the live stack; 4.8 ms per-tick
  overhead @1000 findings), canonical 4-tree suite **642 failed / 197 errors
  vs main baseline 838 — 833 nodeids identical; all 6 extras pass standalone**
  (shared-state flakes — root cause card `6369c03e`: `db_path.parent^3` escapes
  `tmp_path` into shared `/tmp/pytest-of-mana`, a 22-entry cross-session journal
  with a timeline-matched first write; their baseline has 5 unique flakes of
  its own).

- **2026-10-04 · `feat/avion-transcript-batching` lands — transcript writes are
  group-committed.** Card `68ea8b9d`, cherry-pick of `fb7859ced`:
  `AgentConfig.transcript_flush_steps` batches JSONL transcript writes — write
  every step, `flush()` every N (default 1000). **Benchmark (2k steps/scenario,
  quiet window load 1.6, conda)**: group-commit **20,806 steps/s (48.1 µs/step)
  ≈ bare loop (20,621 / 48.5 µs) and +34.4% over per-step flush (15,481 /
  64.6 µs)**; bare and per-step both match the recorded baseline (21.7k/15.7k →
  −5% / −1.4%); callbacks free (21,881); screenshot 19,576. Gates: avion suite
  green (1 isolated load-flake, passes alone) + ruff clean. An earlier run at
  load ~4 showed every row ~2× depressed — record loadavg with any benchmark
  number or the comparison is meaningless.

- **2026-10-04 · `fix/boot-overlay-stall` lands — "boot overlay never
  dismissed" is fixed.** Card `cb089b43`, cherry-pick of `d897399e7` (only the
  overlay commit lands — its branch is stacked on `feat/avion-transcript-batching`,
  which awaits its quiet-window benchmark, see In flight). `StartupOverlay`
  gains a progress-based stall watchdog: `STALL_TIMEOUT_MS = 20_000`, keyed on
  `stage:modelProgress` (mere health ticks don't reset it) → `overlay_timeout`
  event, 600 ms fade, unmount + key-deduped global banner (`startup-degraded`,
  warning, Retry = reload via `useBannerStore`); the ready-path fast exit and
  the 8 s stuck-UI path are unchanged. **Validation**: overlay suite 10/10 (+
  banner/GlobalBanner 16/16), tsc + eslint clean, full web suite **841 files /
  8068 tests green**, firefox ux-flows benchmark **13/13** (baseline was 5/13
  with 8× `boot overlay never dismissed`; now 42 console errors / 0 network) —
  run a worktree stack with `vite --port 3000` + `SLO_WEB_URL=http://localhost:3000`,
  see the CORS gotcha.

- **2026-10-03 · `fix/journey-test-gates` landed (user sign-off).** merge
  `e081ea9f4` (46 files) + stabilization `175d7db8c`: conda-first
  `scripts/python` resolver (`domain.shared.find_server_python`), journey
  module `skipif` server gate + `_web_is_ready`, `hf-*` import tests restored
  (a preserve-commit had regressed them to `kaggle-*`), `SMOKE_ROUTES`
  restored (+28 route smokes), stale `Import`→`Add file` labels, gateway
  `:8080`-aware API assertion, bounded poll-waits for fixed-sleep races.
  Validation (conda + `PYTHONPATH`, load avg 11.6): **journeys 103/103,
  computer_use 21/21, e2e 9/10** — sole red is `config_elements` (training
  config gated behind the guided-setup flow; UX-owner call, drift card). This
  push also carried the avion session's own main-merges (they had merged but
  not pushed).

- **2026-10-03 · cards `13f50db7` `e26dc68c` `24676ff2` `553da7a7` — the avion
  stack lands** (4 branches, carried on shared main by the gates session's
  push):
  - `feat/avion-journeys` (takeover of the dormant journeys card): the 6
    journey commits land; the `test_user_journeys.py` overlap with
    `fix/journey-test-gates` was resolved by keeping the avion port and
    porting all 5 gates deltas into it (`scripts/python` usage, `:5173`
    default + comment, `_web_is_ready` + module skip-guard, 4 training routes
    → 103 tests, DatasetsImport poll already equivalent in avion form).
  - `feat/avion-unify-drivers`: journeys / `run_ux_flows.py` /
    `screenshot_headers.py` drive avion `SyncRunner` instead of ad-hoc
    Playwright (`force=`, `PageControls`); `arken`/`voyager` shims remain.
  - `feat/avion-event-logger`: stdlib-only write-through JSONL journal +
    modular `EventSink` API — `tests/test_stdlib_only.py` proves it never
    imports `domain/`.
  - `feat/avion-agent-loop`: bounded awaits, honest deadline stop, retry only
    the pure-predict path, transcript write safety.
  - **Gates (conda + `PYTHONPATH`)**: avion **366 passed, 1 skipped** (skip =
    the no-websockets degradation test, env-conditional — correct here);
    live journeys **103/103** (avion port validated pre- and post-
    stabilization); ruff clean. **Benchmarks (authoritative env, load <3.5)**:
    agent loop 21.7k steps/s bare / 15.7k with JSONL transcript; event logger
    86.7k ev/s — broken sink isolated at ~journal-only, +5 sinks costs ~5%;
    downcraft e2e sha256 `ok` incl. the zstd-3 row.
  - **Insight → candidate**: residual journey failures that bucket to stale
    selectors are selector drift — a selector-contract probe validating
    journey selectors against the live DOM would fail loudly instead of as
    scattered journey F's.

- **2026-10-03 · card `19cd41dc` — download compression lands in the downcraft
  path.** `feat/downcraft-compression`: decoded byte-space enforced across the
  compressed resume path (Range offsets vs decompressed bytes; wire-space 206s
  no longer corrupt resumes), streamed `resp.read()` + stale-Range duplication
  fixes, SLZ4 resume verifies SHA-256; new
  `scripts/benchmark_downcraft_compression.py`. Gates on the merged tree: 482
  passed + ruff clean + benchmark green.

- **2026-10-02 · card `20260928_075` — full `packages/core-py/tests` no longer
  hangs.** One merge of `feat/infra-thread-teardown` covers
  `fix/pretrain-thread-leak` + `feat/card-hash-chain`:
  - **L1**: pretrain BLAS-thread leak → `os.fork()` deadlock (multimodal dedupe +
    `stop_pretrain` + cancel + conftest teardown).
  - **L3**: idle-manager / fire-and-forget / feedback-workflow / pugqeep stopped
    per test via autouse `_stop_infra_threads` (gated by
    `tests/test_infra_thread_teardown.py`, 9 tests).
  - **faf `shutdown()` fix**: `_STOP` now delivered when the bounded queue is
    full — previously each full-suite session leaked 2 `faf-worker` daemons.
  - **hash-chained kanban ordering**: `chain_index/chain_prev/chain_hash` +
    `verify_chain` (`packages/app-planner`, phase 2).
  - **gate**: chunked resumable runner
    (`~/.cache/slog-journeys/gate_chunks/run_gate.sh`, survives reboots) —
    **683F / 41629P / 0E** vs run7 baseline 594F / 41575P / 141E. Playwright
    error class eliminated (chromium installed into shared
    `~/.cache/ms-playwright`); per-file parity: 82 identical, 19 differing
    (3 converted error-class files + 16 improved).
- **2026-10-02 · local-main backlog published**: health-payload trim
  (10 MB → ~300 B), `:3000` → `:5173` doc claims, doctor Phase A (5 commits
  that sat unpushed).
- **2026-10-03 · journey suites repointed `:3000` → `:5173`** (gate9's
  route-smoke class = stale port; vite lives on `:5173`, card `e47e19ee`
  deferred core-py journey tests). `test_user_journeys`,
  `test_computer_use_training_integration`, `test_e2e_training_trigger` now use
  `BASE = os.environ.get("SLO_WEB_URL") or "http://localhost:5173"` (code edits
  preserved mid-run by the doctor session's `6b74a4024`; results `de2fe6cf4`).
  Live-stack census **95P/11F** (of gate9's 118: 83 green, 11 stale selectors,
  24 no longer exist — pre-merge file collected 99, main's consolidated file
  collects 75).
  Residual 11 = datasets-import dialog/Kaggle ×7, tools selectors ×2,
  Vite-proxy API URL, training config → journey session, card
  `20260929_journey_gates_conda`.
- **2026-10-02 · `fix/doctor-ui` merged** (`40ba857e0`) — /doctor API + page.
- `zzz_test.py` at repo root is a stray from origin's `89e01b58c`
  (shell lazy-loading refactor) — not ours; owner please remove.

- **2026-10-20 · gateway binary provenance closed (card `12aad775`, done).**
  Was partially remediated (Oct 19 rebuild, stamps already clean) but still
  stale vs its own source: `cargo build --release --locked -p slough-gateway`
  in the **shared checkout** recompiled the crate (1m10s → binary 10:50),
  `systemctl --user restart slough-gateway` (PID 171770), and `:8080/health`
  = `{status, gateway:"rust", sidecar{healthy,last_check,model_loaded,
  model_name}, uptime_seconds}` matches this checkout's `HealthResponse`
  (`src/main.rs:146`, handler `:380`; uptime=3s proves the fresh process
  serves it). `strings binary | grep edge-cascade` = 0; foreign `.d` stamps
  = 0. Unit `WorkingDirectory`/`ExecStart` already pointed at the shared
  checkout — no unit edits; edge-cascade worktree stays provenance reference
  only. Journal note (`~/.config/dev-notes/2026-10-01-*.md`) set `status:
  done` by hand — **note the tooling gap: `PlannerStore` reads only
  `.dev-notes/notes.journal.jsonl`, it cannot see the legacy `.md` journal
  AGENTS.md calls source of truth, so CLI `note update` misses those** (the
  `08baf13f` board-sync owner may want this on their radar). From
  `.wt-worldbrain`.

## In flight

- **2026-10-19 · `feat/wire-py` — we own the HTTP wire, phase 1 (card
  `ecea49c8`, stacked on `feat/scene-build`).** `packages/wire-py` is the
  stdlib-only sync framing layer: `wire/framing.py` (request parser —
  version-aware keep-alive, chunked request bodies, TE+CL smuggling guard,
  conflicting-CL rejection, header caps -> 431, malformed -> 400; response
  serializer with wire-owned Content-Length/Connection/Date/Server) and
  `wire/server.py` (`WireServer`: threaded sync accept loop, handler crash ->
  500 + close, desynced connections never reused, prompt `stop()`). TDD: 30
  tests RED -> GREEN, ruff clean; `pytest.ini` adds the package to default
  testpaths/pythonpath (+30 collected; the 24 collection errors are
  diff-identical to HEAD = pre-existing). NEXT: phase 2 ASGI host adapter
  (`wire/asgi_host.py`, asyncio confined to that seam) + SSE; minimal RFC6455
  WS deferred (flip stays gated on WS). uvicorn remains the default host until
  phase 3 gates (`SLO_TRANSPORT` A/B harness + `benchmark_transport.py`) are
  green. **Phase 2 also done (`3eed464dc`)**: `wire/asgi_host.py` (scope +
  per-request loop + streaming frames; crash-before-bytes = 500 + close;
  receive parks post-body - proven against real FastAPI
  `StreamingResponse`, no hang), `serialize_head` extracted, target
  high-byte now a clean 400 (was `UnicodeDecodeError` killing the conn
  thread), `WireServer(asgi_app=...)` mode. 37 tests green; smoke: JSON
  (Content-Length) then SSE (chunked) on the SAME keep-alive connection.
  **Phase 3 part 1 done (`bddedfe6e`)**: `scripts/benchmark_transport.py`
  (raw-socket client for precise TTFB, `--json` for tracking) —
  keep-alive p50 0.115 ms / ~8.3k req/s, fresh-conn p50 0.285 ms,
  8-thread concurrent 5.8k req/s, ASGI adapter p50 0.398 ms
  (+0.28 ms/req = per-request `asyncio.run`), SSE TTFB p50 0.529 ms
  (total 41 ms / 5 frames x 5 ms); TTFB bug fixed pre-commit (absolute
  `perf_counter` had leaked into the latency metric). REMAINING phase 3:
  `SLO_TRANSPORT` wiring + journey gates before any default flip;
  RFC6455 WS deferred (flip gated on it). **Phase 3 lifespan + A/B done
  (2026-10-20, committed on stack tail `feat/npu-caps`)**:
  `wire/asgi_host.py` gains `Lifespan` — one dedicated loop thread drives
  the spec handshake (startup completes before the accept thread spawns,
  shutdown after connections close; failed/crashed/timed-out handshakes
  unwind with no thread leak); `WireServer(lifespan="auto"|"on"|"off")`
  with uvicorn-compatible auto fallback (lifespan-unaware apps keep
  serving). Real FastAPI `lifespan=` interop proven in tests; 48 wire
  tests green (11 new `tests/test_lifespan.py`), ruff clean. A/B: in-process uvicorn 0.53 in `benchmark_transport.py` (same app, same
  raw-socket client, interleaved rows, `--no-uvicorn`) — ours 0.354 ms
  p50 vs uvicorn 41.0 ms tight-loop, localized to uvicorn's split
  head/body writes with no TCP_NODELAY + delayed-ACK (TTFB 0.77 ms,
  body held 40 ms; 100ms-idle capacity row = 0.94 ms) vs our
  single-write framing; adapter overhead +0.23 ms/req; 2 runs recorded
  (note `2026-10-20-wire-layer-phase3-lifespan-ab.md`). REMAINING:
  `SLO_TRANSPORT` wiring + journey gates (shared tree ~110 dirty — the
  window). Next lane: NPU CAPS. From `.wt-worldbrain`.
  **RFC6455 WS done (2026-10-20, committed on stack tail `feat/npu-caps`)
  — the flip gate is CLEARED.** `wire/websocket.py` (sync stdlib:
  handshake validation, 101 + `Sec-WebSocket-Accept`, unmasked server /
  masked client frames, 7/16/64-bit lengths, fragmentation,
  ping→auto-pong) + `wire/asgi_host.py` `run_websocket` (one session loop
  per connection over the ASGI websocket connect/accept/receive/send/
  close/disconnect contract, dedicated single-worker reader executor with
  a shielded in-flight read) + `wire/server.py` routes
  `is_ws_upgrade(request)` → `run_websocket` (session consumes the
  connection, no keep-alive after). Scope grounded on the API's real
  surface — `inference.py` `add_api_websocket_route("/ws/generate",
  ws_generate)` with Starlette accept/receive_text/send_json/close, whose
  interop test pins that exact route shape end-to-end over a real socket.
  Two real bugs caught by RED tests then fixed: (1) cancelled-receive
  teardown deadlock — Starlette's `wait_for(receive,30s)` cancels a
  pending receive and the in-flight read stayed on the default executor
  that `asyncio.run` joins at session end while the connection is still
  open → hang; fixed with a session-owned reader executor torn down
  `shutdown(wait=False,cancel_futures=True)`. (2) reader-close deadlock
  (second order) — `reader.close()` waits on the makefile lock held by
  that blocked read, and `socket.close()` does not interrupt a blocked
  recv (only `shutdown()` does); `_serve_conn` teardown now does
  `conn.shutdown(SHUT_RDWR)` before `reader.close()`. 14 new
  `tests/test_websocket.py` (handshake 101/400/426, text+binary 16/64-bit
  echo, fragmentation, ping→pong, unmasked-client-frame 1002,
  server-push-while-receive-pending, close/EOF, reject→403, plain-HTTP
  untouched, cancelled-receive teardown, FastAPI interop); full wire
  suite **62 green**, ruff clean. pytest gotcha recorded in-file: with
  `from __future__ import annotations` FastAPI resolves the endpoint's
  `websocket: WebSocket` annotation against MODULE globals — importing
  `WebSocket` inside the test silently degrades it to a required query
  param and the route 403-rejects. Benchmark re-run after the teardown
  change (shutdown on every HTTP conn path): no regression (keepalive
  p50 0.087 ms, keepalive_asgi 0.352 ms, uvicorn A/B delta unchanged
  −99.1%). **Tier-1 hardening (uvicorn parity, wire-local)** then closed
  the A/B gap list: proxy-header translation (XFF/XFP → scope
  client/scheme, trusted peers only, `forwarded_allow_ips` default
  `127.0.0.1`), keep-alive idle timeout (`keep_alive_timeout=5.0`,
  read-phase only — SSE writes unbounded, pipelined bytes never
  false-trip), body cap (`max_body` 64 MiB → 413; chunked counts;
  trailers capped), WS message cap (`ws_max_size` 16 MiB on declared
  frame length + fragment assembly → close 1009), `Expect:
  100-continue` (interim only after validation; unknown → 417),
  uvicorn-format access log (`wire.access`, handler + ASGI + WS 101).
  Smuggling guards already existed from phase 1 — the only body gap
  was the cap. 15 new tests (`test_hardening.py` 13 + 2 WS-cap) →
  suite **77 green**, ruff clean; a test-helper flake (discarded
  coalesced pipelined bytes) fixed via shared read buffer (43/60 →
  0/15). Benchmark no regression (keepalive p50 0.099 ms, adapter
  +0.284 ms). Deferred with reasons: body streaming (needs a card),
  permessage-deflate, hot-reload. Only the shared-tree `SLO_TRANSPORT`
  wiring + journey gates remain. From `.wt-worldbrain`.
  **Streaming request bodies (card `c24543a4`, DONE, 2026-10-20)** —
  the last deferred Tier-1 item, landed on `feat/npu-caps`. Bodies over
  `body_buffer_threshold` (default 64 KiB) stream through ASGI
  `receive()` (`http.request` + `more_body`); `max_body` became a
  running total (413 `BodyTooLarge`, matters for chunked); small bodies
  keep the exact single-frame path (hot path pinned by test). New
  `wire/framing.py` `BodyReader` (reads through the connection's
  buffered reader — pipelined bytes never stolen; threshold read-ahead
  re-queued so nothing is lost or reordered at the buffer→stream
  switch), per-request single-worker reader executor in `server.py`
  (the WS seam, `shutdown(wait=False,cancel_futures=True)` in a
  finally), handler mode drains via `read_all()` (buffered contract
  preserved). Two bugs caught by RED tests: (1) **cancellation dropped
  body bytes** — `wait_for(receive())` cancelled the wrapped future
  while the executor read completed, discarding bytes; the WS
  cancel-and-reset pattern would corrupt an HTTP body, so the HTTP side
  uses `asyncio.shield()` (cancellation abandons the await, never the
  future's result); (2) access-log assert race in my own hardening test
  (client has the response before the server thread logs) → bounded
  poll. `_BODILESS` → `_BODYLESS` typo fix. Benchmark gained
  `upload_wire`/`upload_uvicorn` (10 MiB CL, byte-exact per round):
  64 KiB chunks = 3.4× behind uvicorn → `_BODY_READ_CHUNK` 1 MiB =
  **488 MiB/s (1.55×)**; 2 MiB no better → floor is the thread↔loop
  seam itself, core stays sync (execution philosophy). Final record at
  load 15: upload_wire 35.1 ms p50 vs uvicorn 15.6 ms, all other rows
  unchanged (A/B −98.7%). 6 new `tests/test_streaming.py` → suite
  **83 green**, ruff clean. Follow-up on card (low prio): bounded
  read-ahead queue if a bulk-ingest journey demands it.

- **2026-10-19 · `feat/npu-caps` — OS design L2, CAPS + native tensor
  kernels, card DONE (phase 2; `d74e2cef`, stacked on `feat/wire-py`).**
  `IoctlCommand.CAPS` + `TensorDevice.caps()` = the tensor unit's
  machine-readable contract `{name, type, ops, dtypes, backend, parity}`
  — declared once on the ONE TensorDevice whose three faces (CPU tensor
  backend `vm._default_tensor`, kernel `NPUDevice._compute`, runtime
  DeviceManager `"tensor"`) all reach it through the same
  `ioctl()/call()` path; `ops` single-sourced from `list_commands()`
  with a drift test. Phase 2 routes hot ops through the ONE loader:
  `domain/shell/_internal/tensor_kernels.c` (SLO_Q source, gcc+OpenMP /
  NVRTC entries) behind `SLO_TENSOR_NATIVE` (=0 forces the numpy oracle)
  — **benchmark-gated** (`scripts/benchmark_tensor_kernels.py`,
  thread-pinned): softmax 2.5-2.7x, layernorm 3.6-3.8x, rmsnorm 1.4-1.5x,
  gelu 1.5-1.6x shipped; relu 0.97x NOT routed (kernel ships, stays
  engine-tested); matmul stays BLAS. CAPS reports reality: engine
  `native-cpu`, `native_ops = parity = [GELU, LAYER_NORM, RMS_NORM,
  SOFTMAX]` (exact-set ratchet). TDD 19 RED -> GREEN; new
  `test_tensor_native.py` (CPU parity, kill-switch spy, byte-identity
  fallbacks, GPU-vs-oracle on real CUDA); face stash-diff = 12
  pre-existing identical; ruff clean. shas `8050828ae` (routing),
  `e23fdb2d7` (bench), `05df4b22d` (evidence). LANDING: tail of the
  7-branch stack (after `feat/wire-py`). From `.wt-worldbrain`.

- **2026-10-21 · vq-huffman-04 benchmark truth-up (stack tail `feat/npu-caps`,
  `b4fda3edc`; card `vq-huffman-04` DONE).**
  `scripts/benchmark_block_quantization.py` was the card's deliverable all
  along (exact title match) — fixed, ran, recorded: `zlib.crc32` seeds replace
  PYTHONHASHSEED `hash()` (history now comparable), cluster serialize measured
  + **bit-exact integrity-gated** (stale block-only guard; series cards 01-03
  landed `ClusterCodec` streams), median-of-rounds timings, JSON history
  (`benchmark_block_quantization.json`, downcraft precedent). Verdict avg over
  5 realistic NN shapes: **block_q4 5.33x / cos 0.99695 / 0.9 ms encode =
  storage default**, **block_q8 3.20x / MSE 8.39e-09 = fidelity pick**,
  cluster dominated (3.43x, 248 ms encode, ~275x slower) on weight-like data
  — its niche is discrete/heavy-tailed input. Suite gate `-05`: pugqeep/
  compressor 2020 tests **0 fail**; wire-py 83/83; canonical remainder 3784
  tests with 8+2 reds **all stash-proven pre-existing** (drift-baseline lane
  `56b49cf1`/`33773fce`); the 24 core-py collection errors pre-date the fork
  (`merge-base` lacks `domain/shell/_internal/cmds/*` too — main gained them
  after; zero deletions on my side). From `.wt-worldbrain`.

- **2026-10-21 · vq-huffman-06 — `benchmark_vq_approaches.py` truth-up
  (stack tail `feat/npu-caps`, card `202365ec` created+closed this session).**
  The script's "current baseline" was its own local quantile-init `vq_core`
  and it listed adaptive-k + k-means++ as *proposals* — both shipped by series
  card 02. Re-anchored rows on the REAL `PointCompressor`
  (`shipped_vq` + `shipped_vq_k16`), kept `legacy_quantile_k16` as history,
  kept PQ/RVQ/FSQ as the exploration set (documented as-un-Huffman-coded);
  median-of-3 timings, bit-exact decode gate, JSON history. Verdict over 6
  BERT-scale layers: no exploration row beats the shipped codec (PQ 0.71x
  ratio/13.6x MSE, RVQ 0.26x, FSQ 0.81x/28x — FSQ encodes 60x faster, the one
  interesting signal). **Headline: fixed k=16 + the codec's residual fallback
  (accuracy < residual_threshold 0.99, compressor.py:357) dominates adaptive-k
  lossy VQ — 8.22x ratio AND lossless (MSE 3e-21) vs 4.91x lossy.** Cite when
  tuning n_clusters/residual_threshold. From `.wt-worldbrain`.

- **2026-10-21 · board reconciliation — worktree board stale-row graveyard
  cleaned (card `b19e29eb` created+closed this session).** Every unowned todo
  candidate on the fork board was re-pickaxe-verified (10th consecutive
  "continue building" round burning minutes on the same rot rows): **32 cards
  closed with sha-cited evidence** (ancestry-tested against origin/main where
  possible — deliveries like `93286c6fe`/`dd83baf9a`/`b53487b3b`/`3f73ff956`/
  `91e81243b` IN-main; `ca167dbd` proven by origin/main's own `pytest.ini`
  carrying its card id), **10 byte-identical dup-id rows collapsed**
  (`8182563a`×4, `f74b9e75`×3, `90839119`/`a25e490b`/`13aed9ad`/`ec2ae5c1`/
  `7e2af9fe`×2). **Two honesty exceptions: `ccad389e` stays OPEN** — its close
  claims "landed on main 356d73b83" but that sha is only in `chore/*`
  branches, never reached origin/main (note appended); `44c63b91` untouched
  (frontend lane + conflicting board states). Unowned todo: 75 → 38, every
  remaining row intentional (cleanup/lint/ux/My-Note/user-parked/window-
  blocked). Board-only via PlannerStore; NO sync run — fork lacks `cf89ac6a4`
  (the oscillation fix is on a planner branch, not in this fork). From
  `.wt-worldbrain`.

- **2026-10-21 · vq-huffman-07 — compression-knob frontier + the storage-
  accounting double-lie (stack tail `feat/npu-caps`, card `6eaa413c`
  created+closed this session).** Grid adaptive x n_clusters{8..64} x
  residual_threshold{0,0.99,1} on the real PointCompressor
  (`scripts/benchmark_compression_knobs.py`, suite imported from -06). The
  sweep exposed a measurement red flag (ratio identical across thresholds
  while decode flipped lossless) → **two production accounting bugs**:
  (1) `huffman_data` is written only in-memory (`compressor.py:375`) —
  `serialize_params` packs dense uint8 assignments and `from_bytes` never
  rebuilds the tree, so the Huffman compression **dies with the process**;
  (2) `ClusterCodec.nbytes` (`codecs.py:217-232`) counts that phantom
  Huffman (line 225) while omitting the residual serialize packs (line 249).
  Consumers of the lie: `training/executor.py:527`, `pugqeep/library.py:461`,
  `dedup.py:71`, `generic.py:693,696`, `point_interface.py:121`.
  **RETRACTION: the -06 headline "fixed k16 + residual dominates adaptive-k
  (8.22x AND lossless)" was an accounting artifact** — honest (to_bytes)
  ratios: k16+residual = 0.80x lossless, adaptive = 3.84x lossy (0.21x).
  Corrected frontier truths: lossy ratio is ~k-independent (3.8-3.9x — dense
  assigns dominate; k is a fidelity knob), lossless class = 0.79-0.80x raw
  (residual = the price of losslessness), fixed_k64 ≈ adaptive at equal bytes
  (cos 0.99820 vs 0.99776, marginal). benchmark_vq_approaches.py re-anchored
  to to_bytes + fresh history: shipped-beats-PQ/RVQ/legacy HOLDS (legacy 1.01x
  ratio, 13.8x MSE), FSQ flips to **1.04x shipped's honest ratio at 77x
  faster encode**. Storage fix filed as follow-up card (needs approval: nbytes
  truth-up vs persist-Huffman wire-layout change). From `.wt-worldbrain`.

- **2026-10-21 · pugqeep storage truth-up option B — Huffman persists (CLH1)
  + honest nbytes (stack tail `feat/npu-caps`, card `d9028c8f` closed this
  session).** The vq-huffman-07 follow-up landed as the user-chosen option B:
  `ClusterCodec.serialize_params` writes the Huffman bitstream under a
  self-identifying `CLH1` marker (legacy dense blobs keep parsing; the outer
  type code `b"CLH1"` → old readers fail loudly instead of misparsing the
  bitstream as dense assignments), `deserialize_params` rebuilds the tree,
  decodes, and MATERIALIZES dense assignments so every params consumer
  (dedup fingerprints, ratios, lazy slices, to_dict) sees the same shape a
  fresh compression produces — huffman keys restored alongside, so
  re-serialization is idempotent. `nbytes` rewritten to the payload
  contract: residual in EVERY branch; the huffman branch counts the
  persisted bitstream + tree JSON (was: a phantom bitstream counted + a real
  residual omitted); invariants `nbytes <= len(to_bytes())` and
  point-vs-residual-twin delta = exactly `residual.nbytes`. TDD: 11 contract
  tests RED→GREEN (`test_pugqeep_huffman_wire.py`), full pugqeep suite 2151
  green / 3 skipped, ruff clean; 3 stale pins updated with why-comments
  (compressor nbytes formula + the 2 checkpoint `compression_ratio` pins).
  Benchmarks re-recorded post-B (third history record; honest accounting,
  `nbytes_gap` = 306 B per 6-layer config = headers only, under the new
  512-B tripwire): **CLH1 lifts the shipped codec's honest ratio — knobs
  sweep adaptive_k16 3.84x -> 4.50x, fixed_k16_rt0 3.90x -> 7.49x (+92% —
  dense assignments were the dominant cost); vq-approaches shipped_vq
  4.00x -> 5.75x, shipped_vq_k16 (lossless) 0.80x -> 0.90x, while PQ/RVQ/
  FSQ/legacy rows are byte-identical pre/post — the wire change moves
  exactly and only huffman-carrying cluster points.** DISCOVERY needing a user decision:
  `compress_checkpoint`'s default rt=0.99 stores full-size float32 residuals
  on real weights → honest ratio < 1.0 (0.86-0.91 measured) — checkpoint
  "compression" INFLATES the library on disk; the old nbytes omission was
  masking it. Options: lossy checkpoint path (rt=0) vs reframe the stat —
  behavior change, deliberately out of this card's scope. Capacity nuance:
  k=64 clears the 0.99 threshold on all 6 benchmark layers (0 residuals
  fired, genuine compression) — the inflation is a low-k capacity effect.
  From `.wt-worldbrain`.

- **2026-10-04 · startup overlay becomes an ambient boot layer — landed on main
  (`2fb066dce`, card `159300be`) and deployed in `apps/web/dist-vite`.** The
  fullscreen overlay is now `bg-[#0a0a0a]/80` + `backdrop-blur-sm` +
  `pointer-events-none` (its own controls — Retry, Show timing — opt back in
  with `pointer-events-auto`): the shell stays visible **and clickable** for the
  whole boot. The stuck (8 s) and stall (20 s) watchdog screens and the
  key-deduped banner were NOT demoted — they live in the overlay/banner exactly
  as before; only the layer's modality changed. Verified live: dead-core
  walkthrough on `:8082` (t=0.46 s hit-test passes through the overlay to
  `sl-app-content`; t=10.5 s "Still connecting" + Retry `pointer-events:auto`;
  t=21.5 s overlay gone + `GlobalBanner` "Backend not responding") and healthy
  `:8080` (overlay dismissed via ready path, `origins=[localhost:8080]`,
  0 failed requests, 0 console errors). Gates: `StartupOverlay.test.tsx` 11/11
  (incl. pass-through contract), full web suite 8075/8075 @ 841 files, tsc +
  eslint clean. **Deploy = `npm run build:vite` from a checkout ≥ this commit,
  copy `dist-vite/` to the repo root; `ServeDir` reads from disk, no gateway
  restart needed.** From `.wt-orb`/`fix/startup-overlay-busy`.

- **Landed on main (`d949eb509`, card `d484f48a`) and deployed on :8080** — the
  gateway serves the repo root's `apps/web/dist-vite` as its document root;
  rebuild with `npm run build:vite`, then restart `slough-gateway`. From
  `.wt-static`/`feat/static-hosting`. Detail:
  `apps/web/dist-vite` as its **document root** — file hit → asset, browser
  navigation (`Accept: text/html`) → `index.html` (SPA), data request →
  byte-relay; `/docs`, `/redoc`, `/openapi.json` stay proxied (path contract).
  Routing is by request *kind*, not path prefix, because SPA and API share the
  same top-level names. `npm run build:vite` builds **same-origin** (empty
  `NEXT_PUBLIC_API_URL`; `??` not `||` in `lib/config.ts` — a `||` fallback
  would silently restore a second origin). `main.py --web` retired: no Node is
  spawned. **Still open:** the 11 `app/api/**` handlers (9 planner, 1 calendar,
  1 nextauth) have no host in a static build — classification on the card
  before anything is ported or deleted.
- Root-repo session on `fix/startup-finalizers` (supersedes
  `feat/pipe-bounded-execution`); ~40 `feat/*` worktrees active —
  `git branch -vv` + the kanban board name the owners.
- The 683 baseline failures are known drift → card `56b49cf1` (drift baseline).
  Check it before treating a failure as yours.

## Gotchas that cost hours (add yours here)

- **A child log opened with `open(path, "a")` poisons every log-derived
  metric**: `scripts/serve_ui.py` appends to `logs/serve_ui-api.log`, so the
  next reader of `Startup complete — server ready for requests` /
  `Preload import failed` can measure a *previous* run's boot. Hit while
  adding `benchmark_startup.py --via serve_ui`: time-to-ready read as 3.88 s
  off a stale marker (honest value 126.4 s) and the 25→5 s marker diff hid a
  real measurement bug. Fix: give the run its own log dir (the benchmark
  re-points `serve_ui.LOG_DIR`) and delete this run's file before starting.
- **Never force-update `main` to land your commit while another worktree sits
  on it at the old tip**: `git fetch . HEAD:main` / `git branch -f main` are
  refused for good reason — that worktree's next `git commit` re-points the
  branch from *its* HEAD and would silently drop your commit from main's line.
  Land from the worktree that owns `main` (`~/Documents/sloughgpt-api-std`,
  see Sync recipe) after claiming it here first.

- **`PYTHONNOUSERSITE=1` (AGENTS.md) breaks browser suites**: playwright is
  installed in `~/.local/lib/python3.12/site-packages/` (user site), not in
  the conda env — with the flag, `import playwright` dies and every
  Playwright suite fails at fixture setup. Browser suites
  (`test_user_journeys`, `test_computer_use_*`, `test_e2e_*`) currently run
  *without* the flag — green, no `tests`-shadow hit observed for
  `packages/core-py/tests`. Fix = install playwright into the conda env
  (needs user approval), then the flag can apply everywhere.
- `app_planner` editable install (`.pth`) points at the **root repo** copy —
  test a worktree's planner code with
  `PYTHONPATH=<worktree>/packages/app-planner/src`. **Worse: any planner
  command run from a branch that predates the chain work (e.g. the root
  repo before merging `main`) rewrites `board.jsonl` WITHOUT
  `chain_index/chain_prev/chain_hash` and strips the seal.** Chain-aware
  copies re-seal automatically (`sync()` ends in `compute_chains()`); after
  such a write, run a chain-aware `store.sync()` or `git checkout main --
  .kanban/board.jsonl`.
- Chunked-gate collect paths are relative to `packages/core-py` (nearest
  `pytest.ini`) — force-prefix `packages/core-py/`, or 16 colliding basenames
  poison `from conftest import build_test_app`.
- Never nudge a running pytest with SIGUSR1/SIGALRM (unregistered → death);
  diagnose with `-o faulthandler_timeout=150`, unblock exit by reaping
  futex-stuck pool children (`kill -9` children of the pytest PID).
- **Vite 8 dev ignores `define` → for a worktree web server on main's config,
  only origin `:3000` is CORS-safe.** main's `vite.config.ts` never loads
  `.env.local` and its `define` doesn't fire in the Vite 8 dev pipeline (the
  root branch works around it with `inlinePublicEnv` → edge gateway `:8080`,
  `ACAO: *`), so a worktree instance computes base `http://localhost:8000`;
  FastAPI's allowlist (`SLO_CORS_ORIGINS`, default `3000,8000`) then rejects
  `:5175` — measured: 5487 console errors, chat textarea stuck disabled,
  ux-flows collapse to 6/13. Fix: `vite --port 3000` +
  `SLO_WEB_URL=http://localhost:3000` → 13/13.
- **`next lint` no longer exists in this Next version** (parses `lint` as a
  directory) — pre-existing repo breakage; the lint gate is root
  `node_modules/.bin/eslint <changed files>` run directly.
- **`note update`/`note new` auto-run `_auto_sync` → `store.sync()`** — one
  note edit triggers full notes↔board reconcile **plus** the chain reseal, so
  expect a WHOLE-file `board.jsonl` diff (canonical re-order + `chain_hash`
  cascade on dozens of cards) and cards moving to match note statuses. It is
  not a clobber — diff per-card `column` fields before reacting. Two traps:
  sync is non-idempotent (it flips cards `1f24a4f7`/`243016c3` every run —
  revert unintended foreign flips before committing: `git checkout` the
  board, then `move_card(<full-uuid>, <col>)` + `compute_chains()`), and
  `move_card` matches the id **exactly** (an 8-hex prefix returns `False`
  silently; resolve the full uuid first).
- **`.kanban/board.json` is a retired legacy snapshot — the live board is
  `.kanban/board.jsonl`.** They sit in the same dir with the same size class,
  so triaging the queue from `board.json` silently reads a stale card set
  (it showed 10 todo cards vs the canonical 82). The CLI cannot address
  legacy date-style ids either: `board show/move 20260916_100007_…` →
  "Card not found" (it resolves app-planner 8-hex ids). Read the queue with
  `app_planner board ls` or by parsing `board.jsonl`; claim a legacy card by
  `board add`-ing a fresh card that references the legacy id in its
  description (`board add` appends exactly one line — no chain cascade).
