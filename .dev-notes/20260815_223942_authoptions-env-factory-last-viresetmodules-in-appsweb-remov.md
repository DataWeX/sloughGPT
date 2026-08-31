---
id: 20260815_223942_authoptions-env-factory-last-viresetmodules-in-appsweb-remov
title: authOptions env factory — last vi.resetModules in apps/web removed
status: done
tags: web,tests,auth,nextauth,refactor
created: 2026-08-15T22:39:42.153231+00:00
---

authOptions env factory — last vi.resetModules in apps/web removed

CONTEXT: user repeated 'code it properly' after the worker.ts DI round (694786b7). A repo-wide grep found ONE remaining vi.resetModules() outside lib/: apps/web/app/api/auth/[...nextauth]/authOptions.test.ts:15.\n\nROOT CAUSE: authOptions.ts read process.env at module load (githubId/githubSecret/NEXTAUTH_SECRET consts). The test stubbed env per case (vi.stubEnv) then vi.resetModules() + dynamic import so the module would re-read the stubbed env.\n\nFIX (authOptions.ts): exported createAuthOptions(env: NodeJS.ProcessEnv = process.env) that resolves providers and secret from a passed env (resolveProviders/resolveSecret helpers); kept 'export const authOptions = createAuthOptions()' so app/api/auth/[...nextauth]/route.ts (NextAuth(authOptions)) is unchanged. Same behavior for empty-string secret fallback and dev secret.\n\nTESTS (authOptions.test.ts): static import of createAuthOptions; each test builds an env object via makeEnv({...overrides}) (NODE_ENV:'test' default) and calls the factory directly. Dropped vi.resetModules, vi.stubEnv, afterEach(unstubAllEnvs), dynamic import, and the unused mockNextAuth + next-auth mock (type-only import is erased; never asserted). Kept the two provider mocks (github/credentials) which are asserted. 8 tests, plus route.test.ts 2 tests, all pass; npx tsc --noEmit exit 0. Committed 333e923e (2 files, path-scoped).\n\nFINAL STATE: zero vi.resetModules() anywhere in apps/web.