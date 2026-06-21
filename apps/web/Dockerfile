# Build from repository root: docker build -f apps/web/Dockerfile .
FROM node:20-alpine AS base

FROM base AS deps
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json* ./apps/web/
COPY packages/strui ./packages/strui
WORKDIR /app/apps/web
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/apps/web/node_modules ./apps/web/node_modules
COPY --from=deps /app/packages/strui/node_modules ./packages/strui/node_modules
COPY apps/web ./apps/web
COPY packages/strui ./packages/strui
WORKDIR /app/apps/web
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

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
