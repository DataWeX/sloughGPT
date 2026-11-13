# apps/web

Vite SPA. Entry: `vite-entry.tsx`; routes discovered from `app/**/page.tsx` via `vite/routes.ts`. Next.js was removed — Next-style imports must go through `vite/next-compat/` shims (see `scripts/codemod-next-imports.mjs`).

## Loading state is an explicit boolean (2026-10-28)

Loading/streaming UI (dots, spinners, skeletons, cursors) is gated by an explicit boolean — `isStreaming` / `isLoading` / `status === 'loading'` — never inferred from content emptiness. An empty `content` means "nothing to show", not "still loading": a persisted empty message or one cleared by an error must render nothing, not a loading placeholder (`MessageContent.tsx`, mirroring `MessageRenderer.tsx`). Content emptiness (`!content.trim()`) is reserved for validation (send/save disabled) and empty-state display.
