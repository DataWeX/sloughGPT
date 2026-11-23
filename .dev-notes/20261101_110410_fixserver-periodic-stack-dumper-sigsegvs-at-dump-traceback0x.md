---
id: 20261101_110410_fixserver-periodic-stack-dumper-sigsegvs-at-dump-traceback0x
status: done
board: 20261101_110410_fixserver-periodic-stack-dumper-sigsegvs-at-dump-traceback0x
title: "fix(server): periodic stack dumper SIGSEGVs at dump_traceback+0xbc walking other threads' frames"
tags: ["server", "crash", "diagnostics"]
sprint: ""
gh: null
---

# fix(server): periodic stack dumper SIGSEGVs at dump_traceback+0xbc walking other threads' frames

FINDING: the recurring server SIGSEGV is NOT in model code. It is in the periodic stack dumper.

EVIDENCE -- two cores, byte-identical fault site, 16 hours apart:
  Oct 31 19:22 pid 978926 thread 978928  RIP python3.12 + 0x1a3c7b
  Nov 01 11:17 pid 68187  thread 68189   RIP python3.12 + 0x1a3c7b
addr2line + objdump resolve 0x1a3c7b to dump_traceback+0xbc in CPython traceback.c:1259. The disassembly decodes as:
  1a3c76  call   _Py_write_noraise
  1a3c7b  mov    0x70(%r14),%rax      <- SIGSEGV: stale pointer in %r14
  1a3c84  mov    0x8(%rax),%rax       ob_type
  1a3c88  testb  $0x10,0xab(%rax)     Py_TPFLAGS_UNICODE_SUBCLASS  (PyUnicode_Check)
  1a3ca6  call   _Py_DumpASCII
%r14 is a PyCodeObject*, 0x70(%r14) is co_filename, and it then runs PyUnicode_Check on the result. Frame #1 (PyCode_Type) is a .data address, i.e. a garbage unwind, so only RIP is trustworthy.

WHICH KERNEL WAS LIVE at each crash -- this is what rules the kernels out as the direct cause:
  Oct 31 -> numba cache file _fused_attention_multi-558 (Oct 25/29) = the OLD parallel=True + prange(H*seq_len) version
  Nov 01 -> numba cache file _fused_attention_multi-595 (Nov 1 10:42) = the NEW serial rewrite
Both crashed identically. Neither the prange version nor the serial rewrite is the cause; what they share is nogil=True (landed 54c9d300c, Oct 6), which releases the GIL while the kernel runs. Combined with SLO_DUMP_STACKS=1 in the systemd unit (live unit mtime Oct 11), which calls faulthandler.dump_traceback_later(30, repeat=True) and walks EVERY thread's frame chain every 30 s -- precisely the code that trips over a thread parked inside a nogil call.

This also re-explains the red herrings: the dumper kept catching innocent frames (quantization.py summary(), watchdog.py, model_health.py, process_guard.py) because it samples everything. It was the tripwire, not the culprit. Note SLO_DUMP_STACKS is documented in main.py as temporary "drill instrumentation".

USER RULING (2026-11-01): keep the dumper, but stop the cross-thread C walk.

IMPLEMENTATION NOTE -- the literal request is not expressible in the stdlib API. faulthandler.dump_traceback_later() has NO all_threads parameter; it always walks every thread. The only way to dump "current thread only" is faulthandler.dump_traceback(all_threads=False), but called from the timer thread that would dump the timer thread's own idle stack every tick -- zero diagnostic value for hang diagnosis, which is the entire point of the tool.

The implementation that honours the ruling's intent (keep a live multi-thread dumper) while removing the exact crashing instruction is to walk the threads in PYTHON via sys._current_frames(), which returns frame objects the caller HOLDS STRONG REFERENCES to. That removes faulthandler's raw C-level co_filename deref -- the precise instruction at 0x1a3c7b -- while still showing every thread.

This doubles as a TEST OF THE HYPOTHESIS: if SIGSEGV persists after the switch, that proves genuine heap corruption in the kernels rather than a faulthandler walk defect, and points the investigation straight back at card for the prefill attention work.

OWNED PATH: apps/api/server/main.py (_install_stack_dump_timer at line 234, _cancel_stack_dump_timer at line 259). Preserve the SLO_DUMP_STACKS / SLO_DUMP_STACKS_INTERVAL env contract and the stderr destination; systemd journal captures stderr. Stale tests/server/test_stack_dump_timer.cpython-312-pytest-9.1.1.pyc exists with no .py source -- do not resurrect it, just do not leave a new .pyc.

VERIFY: with SLO_DUMP_STACKS=1 and SLO_DUMP_STACKS_INTERVAL=2 the server logs a dump every 2 s showing multiple real threads (the uvicorn event loop, the pool, watchdog) not just the dumper; the dump output still reaches journalctl --user -u slough-api; and no SIGSEGV occurs across a soak with concurrent inference requests.

CLOSED 2026-11-03: fixed by 392fa637a (GIL-safe stack-dump thread, sys._current_frames walk) — confirmed ancestor of main; done-card 20261102_020229 tracks the verification.
