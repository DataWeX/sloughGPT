# Build from the **repository root** so `file:../../packages/strui` in package.json resolves.
# Example: `docker build -f apps/web/Dockerfile .`
# Compose: `infra/docker/docker-compose.yml` uses `context: ../..` and `dockerfile: apps/web/Dockerfile`.

FROM node:20-alpine AS base

FROM base AS builder
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json* ./apps/web/
COPY packages/strui ./packages/strui
COPY apps/web ./apps/web
WORKDIR /app/apps/web
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm ci && npm run build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/apps/web/public ./public
COPY --from=builder /app/apps/web/.next/standalone ./
COPY --from=builder /app/apps/web/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT=3000

CMD ["node", "server.js"]
