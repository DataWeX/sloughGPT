/**
 * Shared UI helpers.
 *
 * `cn` lives in `packages/strui/src/lib/cn.ts` — the one shared copy that
 * apps, packages and worktrees all reuse. `apps/web` previously carried a
 * byte-identical local duplicate (same clsx + tailwind-merge body), so this
 * module now re-exports the shared implementation rather than copying it.
 *
 * The deep `./lib/cn` subpath is deliberate: the strui root barrel also
 * exports 85 symbols including every React component, which node-environment
 * tests have no business pulling in.
 *
 * Existing `@/lib/utils` imports keep working; importing `@sloughgpt/strui`
 * directly is equivalent — `apps/web/lib/cn.test.ts` covers the same code.
 */
export { cn } from '@sloughgpt/strui/lib/cn'
