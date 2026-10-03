/**
 * Global v86 x86 emulator — loaded at runtime from the vendored public asset
 * /v86/libv86.js (UMD build sets window.V86, classes typed loosely here).
 * It is deliberately NOT imported from the npm package: the package ESM
 * references node builtins (fs, crypto, perf_hooks) inside node-only branches
 * that break eager bundle resolvers (Turbopack). See lib/v86-controller.ts.
 */
interface Window {
  V86?: new (options: Record<string, unknown>) => unknown
}
