/**
 * Browser stub for the v86 package.
 *
 * The v86 ESM build references node builtins (`fs`, `perf_hooks`) inside
 * node-only branches that never execute in the browser, but Turbopack still
 * tries to resolve them at build time and fails. Webpack handles this via
 * `externals: { v86: 'v86' }` in next.config.js, but `next dev` runs on
 * Turbopack where the `webpack` key is ignored — so this stub is referenced
 * from `turbopack.resolveAlias` (`v86: { browser: ... }`) instead. It only
 * surfaces if the in-browser VM is actually initialized. Never import
 * directly.
 */
const stub: Record<string, never> = {}
export default stub
