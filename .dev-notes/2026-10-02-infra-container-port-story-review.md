---
id: 20261002_infra_container_port_story
title: "Infra: container-port story review — compose 3000:3000 vs live vite :5173 (deferred from card e47e19ee)"
status: done
date: 2026-10-02
resolved: 2026-10-11
branch: fix/container-port-story
landed: 682fcb4fc
tags: [infra, docker, ports, followup]
board_id: 257d310b-4143-4818-a746-10bbe3b81d73
board: 257d310b-4143-4818-a746-10bbe3b81d73
hash: ecf25e138488b7055cb7d20a3f4b41c7fb4c5e6a4b5d209ab06a114483d04f99
---
# Infra: container-port story review — compose `3000:3000` vs live vite :5173

## CONTEXT

Card e47e19ee (done, merged 34898f48) retired stale WEB :3000 claims from
scripts/docs but deliberately excluded container topology. Remaining :3000 refs:

- infra/docker/docker-compose.yml (`3000:3000` publish + wget healthcheck)
- docker-compose.prod.yml (two wget healthchecks on :3000)
- nginx.prod.conf (`server web:3000` upstream)
- infra/docker/.env.example (NEXTAUTH_URL)
- infra/docker/deploy.sh (Grafana line — probably Grafana's own port, low priority)
- apps/web/Dockerfile.vite:36 (`wget 127.0.0.1:3000/health` container-internal)
- infra/scripts/deploy.sh:56 (`curl localhost:3000` web health check)

## TASK

Determine the truth — does the web container actually listen on :3000 INSIDE the
container (container-internal port independent of the host's live vite :5173)?

- If yes: compose/nginx are valid and only the host-facing claims were stale.
- If no: healthchecks/upstreams are dead and must be retargeted.

Then fix accordingly (or document the port story in docs/engineering-overview.md
if valid).

## ACCEPTANCE

- docker-compose config validated (`docker compose config -q` or equivalent)
- healthchecks point at a real listener
- no behavior change to the live systemd stack (slough-web :5173 etc. untouched)

## SHARED WORKSPACE

- index updated = no new app/package/doc (correction to existing
  engineering-overview if needed)
- shared copy reused = root toolchain, no installs
- coordination named = card e47e19ee (done, source of deferral)

---

## RESOLVED 2026-10-11 — landed `682fcb4fc` on `fix/container-port-story`

**VERDICT: yes.** The web container does listen on `:3000` inside the container,
so compose/nginx are valid and nothing is retargeted.

The container is **not** a Node server. `apps/web/Dockerfile.vite` builds the
static SPA (`npm run build:vite` → `dist-vite/`) and serves it from
`nginx:1.27-alpine` with `apps/web/nginx.conf`, which is `listen 3000;`. Vite is
never consulted for a port — `vite.config.ts` sets no `server.port`, so the live
dev server takes Vite's default **5173**. Two processes, two ports, both correct.

Every `:3000` ref above checks out: `3000:3000` publish + `EXPOSE 3000`; the
Dockerfile healthcheck (nginx answers `/health` with 200); prod's
`localhost:3000` and `localhost:3000/api/health` (nginx strips the `/api`
prefix, so it lands on the sidecar's `/health` = `routers.health`);
`nginx.prod.conf` upstream `server web:3000`.

**Correction to this note's list:** `infra/scripts/deploy.sh:56`
`curl localhost:3000` was filed as a possibly-stale host-facing claim. It is
not — that script drives `docker-compose.prod.yml`, so it is a *container*
check. Identifying which deploy script is which turned a suspected defect into a
non-defect.

**Real defect found and corrected:** `.env.example` claimed NextAuth was
"required for web service". It is not — the service is a static SPA with no
auth, `apps/web` has no `next` dep and no `next.config.*`, and the only
consumers of `NEXTAUTH_SECRET`/`NEXTAUTH_URL` are orphaned App Router files
under `apps/web/app/api/auth/[...nextauth]/` that no build produces. Comment
corrected; the vars stay set and `docker-compose.prod.yml:137` still injects
`NEXTAUTH_URL`, because dropping an injected variable is a container-stack
behaviour change rather than a doc fix.

The Grafana line (`infra/docker/deploy.sh:185`) is left alone: Grafana's own
default port *is* 3000, so the line is correct and only reads as confusing next
to the web service.

**Gates:** `tests/test_shared_indexes.py` 6 passed; both compose files parse as
YAML (docker not installed here, so this is the equivalent of
`docker compose config -q`, cross-checked against nginx.conf's actual `listen`
rather than trusted from compose); `+46/−2`, zero behaviour change — the compose
files are untouched, so the live systemd stack is unaffected by construction.
