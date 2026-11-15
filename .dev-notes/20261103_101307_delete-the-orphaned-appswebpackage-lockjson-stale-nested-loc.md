---
id: 20261103_101307_delete-the-orphaned-appswebpackage-lockjson-stale-nested-loc
status: done
board: 20261103_101307_delete-the-orphaned-appswebpackage-lockjson-stale-nested-loc
title: "Delete the orphaned apps/web/package-lock.json (stale nested lock, nothing references it after 8d7191279)"
tags: ["ci", "build"]
sprint: ""
gh: null
---

# Delete the orphaned apps/web/package-lock.json (stale nested lock, nothing references it after 8d7191279)

FOLLOW-UP to 8d7191279 (card 20261103_084324). That fix moved every consumer to the ROOT workspace lock: apps/web/Dockerfile deps stage now copies root package.json + package-lock.json + the four workspace manifests, and cache-dependency-path in cloudflare-pages.yml + deploy-git-sync.yml was repointed to package-lock.json.

Verified 2026-11-03: grep over .github/workflows/*.yml + apps/web/Dockerfile finds ZERO references to apps/web/package-lock.json anymore. It still pins react 18.3.1 / no @react-three/fiber (the ERESOLVE source), so it is a trap for anyone who runs npm ci inside apps/web with --workspaces=false.

SCOPE: delete apps/web/package-lock.json only, and grep to confirm no script/docs tell humans to regenerate it (supersedes card 20261101_114838, whose prescribed fix -- regenerating the nested lock -- is now obsolete: the nested install path no longer exists). Optionally add apps/web/package-lock.json to .gitignore so it cannot come back from a local --workspaces=false run.

GATES: web test job (npm ci at workspace root) unaffected; docker build job (build) is the real verifier.
