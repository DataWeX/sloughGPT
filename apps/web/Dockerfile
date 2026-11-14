# Vite static SPA image.
# Build from repository root:
#   docker build -f apps/web/Dockerfile -t sloughgpt/web .
# Optional: --build-arg API_BASE=http://api:8000  (default for compose)

FROM node:20-alpine AS base

FROM base AS deps
WORKDIR /app
# One lock, one truth: install from the WORKSPACE ROOT lock, the same install
# every CI job uses (npm ci inside a workspace member climbs to the root
# anyway). The nested apps/web lock is stale — react ^19 + r3f landed in
# package.json (fd86ab831) without regenerating it, so the old nested
# `npm ci` died on ERESOLVE (react 18.3.1 vs @react-three/fiber peer >=19).
# All four workspace manifests are required: npm ci validates every
# workspace entry against the lock before installing.
# --ignore-scripts: matches test-web (husky prepare needs .git; cypress's
# postinstall downloads a browser binary the image never runs).
COPY package.json package-lock.json ./
COPY apps/web/package.json ./apps/web/package.json
COPY packages/strui/package.json ./packages/strui/package.json
COPY packages/sdk-ts/typescript-sdk/package.json ./packages/sdk-ts/typescript-sdk/package.json
COPY packages/eslint-config/package.json ./packages/eslint-config/package.json
RUN npm ci --ignore-scripts

FROM base AS builder
WORKDIR /app
# The root install hoists all deps to /app/node_modules and links
# @sloughgpt/strui as a workspace symlink; sources layer on top. Copy the
# whole deps tree (COPY of a member node_modules that npm never created
# would fail the build) and let source COPYs merge over it — .dockerignore
# keeps any host node_modules out of the context, so the installed tree wins.
COPY --from=deps /app ./
COPY apps/web ./apps/web
# apps/web/lib/* is a facade: http-client/dev-log/sse-client re-export
# ../../lib/* (the shared apps/lib implementation) — the image needs it.
COPY apps/lib ./apps/lib
COPY packages/strui ./packages/strui
# apps/web/tsconfig.json extends the repo-root base config — vite v8/rolldown
# resolves it during the build and hard-fails without it.
COPY tsconfig.base.json ./tsconfig.base.json
WORKDIR /app/apps/web
RUN npm run build:vite \
  && test -f dist-vite/index.html

FROM nginx:1.27-alpine AS runner
ARG API_BASE=http://api:8000
# nginx.conf proxies /api/ to localhost:8000 by default (host static serve);
# in Docker the API is the compose service `api`.
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
RUN sed -i "s|http://localhost:8000|${API_BASE}|g" /etc/nginx/conf.d/default.conf
COPY --from=builder /app/apps/web/dist-vite /usr/share/nginx/html
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:3000/health || exit 1
