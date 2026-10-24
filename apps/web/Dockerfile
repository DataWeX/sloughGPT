# Build from repository root: docker build -f apps/web/Dockerfile .
FROM node:20-alpine AS base

FROM base AS deps
WORKDIR /app
# Workspace-root install: root package-lock.json is the ONE dependency tree CI
# tests (npm workspaces — member installs are a shadow tree nobody validates;
# the old standalone apps/web install could not resolve plugin-react/vite/vitest
# peers). All four member manifests must be present for `npm ci` to accept the
# workspace layout.
COPY package.json package-lock.json ./
COPY apps/web/package.json ./apps/web/
COPY packages/strui/package.json ./packages/strui/
COPY packages/sdk-ts/typescript-sdk/package.json ./packages/sdk-ts/typescript-sdk/
COPY packages/eslint-config/package.json ./packages/eslint-config/
RUN npm ci

FROM base AS builder
WORKDIR /app
# Carry the whole installed workspace (root node_modules + nested per-member
# entries such as apps/web/node_modules/typescript@5.9.3), then overlay sources.
COPY --from=deps /app ./
COPY tsconfig.base.json ./
COPY apps/web ./apps/web
# apps/lib is imported by apps/web via relative paths (http-client -> ../../lib).
COPY apps/lib ./apps/lib
COPY packages/strui ./packages/strui
WORKDIR /app/apps/web
ENV NEXT_TELEMETRY_DISABLED=1
# next.config.js only enables output:'standalone' when NODE_ENV=production at
# config-load time; the runner stage COPYs .next/standalone. CI runs the same
# command (npx next build --webpack) so the artifact graph matches the tested one.
# With outputFileTracingRoot = repo root, standalone mirrors the workspace:
# server.js lands at .next/standalone/apps/web/server.js and resolves public/
# .next/static relative to its own dir (__dirname) at runtime.
RUN NODE_ENV=production npx next build --webpack \
  && test -f .next/standalone/apps/web/server.js \
  && test -d .next/static

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# standalone mirrors the workspace (outputFileTracingRoot = repo root):
# server.js sits at apps/web/server.js and resolves public/ + .next/static
# relative to its own directory at runtime.
COPY --from=builder /app/apps/web/.next/standalone ./
COPY --from=builder /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder /app/apps/web/public ./apps/web/public

USER nextjs

EXPOSE 3000
ENV PORT=3000

CMD ["node", "apps/web/server.js"]
