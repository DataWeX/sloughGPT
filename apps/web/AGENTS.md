<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

## Loading state is an explicit boolean (2026-10-28)

Loading/streaming UI (dots, spinners, skeletons, cursors) is gated by an explicit boolean — `isStreaming` / `isLoading` / `status === 'loading'` — never inferred from content emptiness. An empty `content` means "nothing to show", not "still loading": a persisted empty message or one cleared by an error must render nothing, not a loading placeholder (`MessageContent.tsx`, mirroring `MessageRenderer.tsx`). Content emptiness (`!content.trim()`) is reserved for validation (send/save disabled) and empty-state display.
