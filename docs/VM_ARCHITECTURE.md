# VM Stacks — Architecture Review (2026-10-26)

Card `20261026_092427_vm-overhaul-c` (umbrella `20261026_085725`). Evidence-based
review of every "VM-ish" stack in the repo: what each is for, what merges, what
retires, and how fault containment holds. **No code changed by this record** —
each ruling below is emitted as a subcard (or joined to an existing card) so the
work gets its own gates.

Input: card A's evidence sweep (`20261026_092426`, console corruption root
causes) and cards B/D's landing (`9b0da882d`, `e96b47475`).

---

## 1. The four stacks (as they actually are)

The umbrella card named them "own-VM console, x86 QEMU VM, v86 browser Linux,
Dait/Buildroot". The survey corrects two of those names — the corrections are
themselves findings.

### S1 — Dait shell console (own-VM console) · the product surface

In-process Python REPL for operating the local AI backend. Singleton
`ShellREPL` (`domain/shell/_internal/repl.py`) drives `DaitRuntime`
(`runtime.py`) over a **simulated** kernel (`kernel.py` — `SyscallNumber`
FORK/MALLOC/TRAIN_START are Python objects, not real syscalls). Not an OS
process, not a hardware VM.

Journeys: operator types commands to manage models/souls/training — CLI
`sloughgpt shell` (`docs/SHELL.md`), HTTP `POST /shell/exec{,/stream}`
(`apps/api/server/routers/shell.py`), web `/shell` pane
(`TerminalPanel.tsx` + view-seam VT engine `apps/web/lib/terminal/engine.ts`).
I/O is a swappable seam (`ConsoleIO`/`MemoryIO`); the wire is sanitized and the
stream is genuinely incremental since card B (`9b0da882d`).

**Maturity: high.** This is the canonical interactive surface. Cards A/B fixed
its data path; card D's RetroBIOS language lands on it next.

### S2 — x86 interpreter VM ("x86 QEMU VM") · educational substrate

**There is no QEMU anywhere in this repo** (no QEMU code, config, or doc — see
§5 for the exact grep). The card's "x86 QEMU VM" is `X86VirtualSystem`
(`domain/shell/_internal/vm.py`) + `VMEngine` (`vm_engine.py`): a pure-Python
x86-32 interpreter with INT 0x80 syscalls, RBAC roles (`vm_permissions.py`),
a device table (`docs/vm-devices-spec.md`), and a training bridge
(`vm_training_bridge.py` → `/training` API). Surfaced via `/vm/run`
(`routers/vm.py`), CLI `sloughgpt vm run|debug`, and the `/vm` web page
(`docs/VM_CONSOLE.md`).

Journey: learn/inspect x86 + RBAC-gated training from assembly. It never boots
a Linux image and never will at this maturity — it is an interpreter, not a
machine emulator.

**Maturity: medium** (solid tests + docs; distinct journey, keep).

### S3 — v86 browser Linux · real-Linux-in-browser substrate

Real x86 Linux running as WASM **client-side** (`v86` npm package), no server
VM. `/vm` page → Browser VM tab; local-first disk (`buildroot/buildroot.img`)
with CDN fallback (`copy.sh`); vendored wasm/BIOS under `public/v86/`,
`public/bios/`; `useV86.ts` + `V86Controller` + `V86TerminalPanel.tsx`.

**Known gap:** main's loader carries the "this.V86Class is not a constructor"
bug (npm stub alias); the fixed port (vendored `libv86.js` UMD, ISO/GUI
gallery, 57 tests) lives **unmerged** on `feat/v86-tiny-linux-port` (worktree
kind-pixel, `b9484b1ee`). Console-corruption reports against the Browser VM
tab are this bug class, not S1's (S1's class was fixed in card B).

**Maturity: medium** (port branch ahead of main; render path still ad-hoc).

### S4 — Buildroot image factory · S3's disk builder (and a naming trap)

`buildroot/` builds a minimal Linux 6.1 + BusyBox image **for S3 only**
(`build.sh` → `/buildroot/buildroot.img`; `useV86` prefers it, CDN fallback).
Overlay `S99dait` starts placeholder `dait-kernel`/`dait-shell`/`dait-api`
binaries (`packages/sloughgpt/sloughgpt.mk`) — an aspirational bridge of S1's
services into the guest, never wired.

**Naming collision (finding):** "Dait kernel" means two different things —
S1's in-process Python `Kernel`, and S4's placeholder guest binaries. One name
per thing; the review rules the Python one keeps "Dait" (it is the product
kernel; S4's binaries are guest services).

**Maturity: low-medium** (image builds; the dait-guest bridge is a stub).

---

## 2. Rulings — merge / retire / keep

| #   | Ruling                                                                                                                   | Disposition            | Rationale                                                                            |
| --- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------------------------------ |
| R1  | **S1 keeps the name "Dait kernel"; S4's guest binaries drop it** (rename to guest-side names in `sloughgpt.mk`/overlays) | keep S1, rename S4     | one name per thing; S1 is the product kernel                                         |
| R2  | **Retire the "x86 QEMU VM" name** — docs/cards say "x86 interpreter VM (X86VirtualSystem)"                               | docs-only rename       | zero QEMU in tree; the name misdirected this review's own scope                      |
| R3  | **Land `feat/v86-tiny-linux-port`** onto main                                                                            | merge (subcard C1)     | fixes a live constructor bug; 57 tests; ISO/GUI work already paid for                |
| R4  | **S2 stays** as a distinct substrate (assembly/RBAC/training bridge) — no merge into S1, no real-QEMU introduction       | keep                   | distinct journey; interpreter ≠ shell; v86 already covers "real Linux"               |
| R5  | **S3 render path adopts S1's view-seam VT engine** (`engine.ts`) over time — one terminal interpreter for every pane     | follow-up (subcard C2) | card D's engine is the shared seam; drift between panels caused the corruption class |
| R6  | **S4 stays** as S3's image factory; the S99dait dait-guest bridge is parked until a journey demands guest services       | keep, bridge parked    | no journey today; placeholder binaries would rot silently                            |

**One story:** S1 is the product console (fix + design-language lane: B landed,
D in flight). S2 is an educational interpreter beside it. S3 is the browser's
real-Linux window, built by S4's image factory. They share a console design
language (card D) and, from R5, a terminal engine — they do not share an
execution substrate, and should not.

---

## 3. Fault containment (Tier-1: shared root must not take the system down)

The avion/arken incident's rule, applied here. What actually shares a failure
domain today:

1. **One process-wide `ShellREPL`** — serialized by `_exec_lock` (card B),
   runs via `asyncio.to_thread` inside the API process. A wedge is a stuck
   thread, not a dead server — but a hard crash (segfault-class bug in a
   kernel path) **is** an API-process crash. Accepted for S1 at current
   maturity; revisit when the shell gains untrusted-input automation.
2. **S2 runs in-process too** (`/vm/run` → interpreter in a worker thread).
   Same containment story as S1; RBAC gates entry, interpreter bugs do not
   get a sandbox boundary. Subcard C3 evaluates process isolation for
   `/vm/run` (it executes guest-authored instruction streams — the highest
   untrusted-input risk of the four).
3. **S3 is already contained** — it runs in the user's browser tab. The
   server cannot be taken down by v86; the tab can. No action.
4. **S4 is build-time only** — no runtime failure domain. No action.
5. **Cross-surface blast radius** — `/shell` and `/vm` routers are separate
   contracts over the shared process (descriptor-projected, per the endpoint
   rule). A future move of S1/S2 execution into a worker process must keep
   the seam explicit (sync core / async host) — never async-ify the kernel
   to fix containment (AGENTS.md execution philosophy).

The existing modular Tier-1 card (`feat(modular): fault containment Tier-1 —
shared root can't take the system down`) owns the general policy; subcard C3
is its VM-specific projection.

---

## 4. Subcards (emitted with this record)

| Card   | Scope                                                                                     | Owner paths                                                                                                         |
| ------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **C1** | Land `feat/v86-tiny-linux-port` on main (rebase, gates, merge)                            | `apps/web/hooks/useV86.ts`, `V86TerminalPanel.tsx`, `lib/v86-controller.ts`, `public/v86/`                          |
| **C2** | v86 pane adopts `lib/terminal/engine.ts` at the view seam (raw tier; fallback plain)      | `V86TerminalPanel.tsx`, `useV86.ts`, engine consumers                                                               |
| **C3** | `/vm/run` process-isolation evaluation → decision + card (pairs with modular Tier-1 card) | `routers/vm.py`, `domain/shell/_internal/vm.py` (eval only, no code in first pass)                                  |
| **C4** | Naming truth-up: R1 guest-binary rename + R2 "QEMU"→"interpreter" sweep in docs/cards     | `packages/sloughgpt/sloughgpt.mk`, `buildroot/` overlays, `docs/VM_CONSOLE.md`, `docs/vm-devices-spec.md`, this doc |

Subcards land on the board under umbrella `20261026_085725`; D (RetroBIOS
design language) continues independently — R5 makes it the shared-render
precondition for C2.

## 5. Verification for this record

- Zero-QEMU claim: `grep -ri qemu docs/ domain/ scripts/ apps/` → only this
  record plus one base64 substring in `apps/mobile/package-lock.json`
  (2026-10-26, main `af5d3281f`) — no QEMU code, config, or doc exists.
- Engine on main: `apps/web/lib/terminal/engine.ts` present at `e96b47475`.
- Port branch ahead: `origin/feat/v86-tiny-linux-port` head `ff722715f`
  contains `b9484b1ee` (constructor fix); main does not.
- S1/S4 naming: `grep -rn 'dait-kernel\|dait-shell' buildroot/ packages/sloughgpt/`.
