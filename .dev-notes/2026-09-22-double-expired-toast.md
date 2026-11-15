# Double toast: "Previous training session expired"

**Status:** done
**Sprint:** —
**gh:** —

## Task
`useTrainingSession` mounts twice (Sidebar `TrainingIndicator` + training page /
desktop+drawer), each runs `reconcile()`, each calls
`addToast('Previous training session expired — server was restarted')`.
Toast store had no dedupe → two identical toasts.

## Body
1. `toast-store.addToast`: if same `message`+`type` already visible, return
   existing id (no second toast).
2. Regression: concurrent double mount of `useTrainingSession` with stale shell
   + idle server → exactly one expired toast.

## Done
- Branch `feat/fix-double-expired-toast` commit `bb283e6d`, cherry-picked to main as `caa16d1f`.
- Board card `3fcdde3d` marked done on main (`423a70c9`).
- Verified: 63 vitest green (toast-store + useTrainingSession + DataStep etc.), tsc 0.

## Constraints
- Same message may reappear after the 6s auto-dismiss window.
- No reconcile single-flight yet (per-instance poll refs) — toast dedupe only.
