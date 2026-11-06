# SloughGPT Web UI

Modern TypeScript-based web interface for SloughGPT using `@sloughgpt/strui` components.

## Features

- 💬 **Chat Interface** - Real-time conversations with AI models
- 📊 **Dataset Management** - Create, view, and manage training datasets
- 🤖 **Model Management** - Browse and configure available AI models
- 📈 **Monitoring Dashboard** - Real-time system metrics and health checks
- 🔄 **Real-time Updates** - Live data with WebSocket support
- 📱 **Progressive Web App** - Installable with offline capabilities

## Tech Stack

- **Framework**: Vite + React 18 SPA (`react-router-dom`), TypeScript
- **Routing**: pages discovered from `app/**/page.tsx` by `vite/routes.ts`; entry `vite-entry.tsx`
- **UI Library**: `@sloughgpt/strui` (workspace package `packages/strui`)
- **State Management**: Zustand
- **Charts**: Recharts
- **Styling**: Tailwind CSS

## Getting Started

### Prerequisites

- **Node.js 20** (match repo root **`.nvmrc`**; CI uses the same major)
- npm or yarn

### Installation

```bash
# From repository root — the web app lives under apps/web
cd apps/web

# Install dependencies
npm install

# Or with yarn
yarn install
```

### Development

```bash
# Start development server
npm run dev

# The app will open at http://localhost:5173
```

Before pushing changes, run the same checks as CI: **`npm ci && npm run ci`** (from this directory — lint, typecheck, Vitest, then **`npm run build`**, i.e. the **Vite** production build into **`dist-vite/`**; same as job **`test-web`**, which installs once at the repository root with **`npm ci --ignore-scripts`** and then runs **`eslint`**, **`tsc --noEmit`**, **`vitest run`**, **`npm run build`** in this directory).

**Talking to models:** set **`NEXT_PUBLIC_API_URL`** to your FastAPI base (default `http://localhost:8000`). Use **Models** to **`POST /models/load`** (`model_id` in JSON), then **Chat** sends message history to **`POST /chat`** (SSE — the endpoint's only state; a caller wanting a complete reply folds the stream itself). Older single-prompt paths **`/infer/generate/stream`** / **`/infer/generate`** remain available for other clients. The client **`api.loadModel`** matches that contract (not `/models/{id}/load`).

**UI vs core engine:** this app is **only** HTML/CSS/TS and `fetch` to the API. It does **not** bundle or import Python (`packages/core-py`, trainers, or inference kernels). Deploy the static Vite build on any static/hosted frontend; run the FastAPI process separately — the only link is **`NEXT_PUBLIC_API_URL`** and the JSON contracts in **`lib/api.ts`** (see also **`docs/STRUCTURE.md`**).

**Cypress E2E** (mocked FastAPI, no Python process): **`npm run e2e:vite`** starts the Vite dev server on **:5173** via `start-server-and-test` and runs Cypress headless against it. Or start `npm run dev` yourself and use **`npm run e2e`** / **`npm run e2e:open`** (baseUrl `http://localhost:5173`). **`npm run ci:e2e`** runs **`npm run build`** then **`e2e:vite`**. Intercepts target the API origin the client was built with — `cypress.config.ts` reads **`NEXT_PUBLIC_API_URL`** with the same `loadEnv()` call as `vite.config.ts`.

### Build for Production

The production site is a **static build served by the gateway** — one origin
for shell, API and SSE (no Node runtime in the serving path, no CORS):

```bash
npm run build:vite     # → dist-vite/ , built same-origin (empty API base)
```

With `MAN_STATIC_DIR` defaulting to `apps/web/dist-vite` when it exists, the
gateway serves it as its **document root**: file hit → asset, browser
navigation (`Accept: text/html`) → `index.html` (SPA owns the routes), data
request (`application/json` / `*/*` / `text/event-stream`) → byte-relay to the
sidecar. The shell therefore loads even while the API is down. Restart the
gateway to pick up a fresh build.

`npm run build` (plain `vite build`) and `npm run start` (`vite preview`) are the
local production path; **`npm run build:vite`** is that same build with
**`NEXT_PUBLIC_API_URL=`** forced empty, which is what a same-origin gateway
deploy wants.

### Docker image

The **`file:../../packages/strui`** dependency must resolve, so build the image from the **repository root**:

```bash
docker build -f apps/web/Dockerfile -t sloughgpt/web:latest .
```

**`infra/docker/docker-compose.yml`** `web` service uses `context: ../..` and `dockerfile: apps/web/Dockerfile` (builder runs **`npm run build:vite`**, the nginx stage copies **`dist-vite/`**). Root **`.dockerignore`** excludes **`dist-vite/`**, **`.next/`** and **`node_modules/`** so local caches are not copied into the build context.

## API Configuration

The web UI connects to the FastAPI backend. By default, it expects the API at `http://localhost:8000`.

To change the API URL, copy **`.env.example`** to **`.env.local`** (or edit **`.env`**) and set **`NEXT_PUBLIC_API_URL`** — see **`lib/config.ts`** (`http://localhost:8000` by default).

## Available Scripts

| Script                     | Description                                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm run dev`              | Start the **Vite** dev server on **:5173** — matches **`slough-web.service`** (the live stack)                                               |
| `npm run dev:clean`        | **`npm run clean`** (remove **`dist-vite/`**) then `vite`                                                                                    |
| `npm run dev:restart`      | Graft port **5173** via **`../../scripts/grab-port.sh`**, then `npm run dev`                                                                 |
| `npm run build`            | **Vite** production build → **`dist-vite/`** (`MAN_STATIC_DIR` defaults there when present)                                                  |
| `npm run build:vite`       | Same build with **`NEXT_PUBLIC_API_URL=`** forced empty — the same-origin variant the **Dockerfile** uses                                    |
| `npm run start`            | **`vite preview`** — serve the production build                                                                                              |
| `npm run lint`             | ESLint over the app (**`eslint . --max-warnings 900`**; build dirs excluded via **`.eslintignore`**)                                         |
| `npm run typecheck`        | TypeScript `tsc --noEmit`                                                                                                                    |
| `npm run test`             | Vitest unit tests (**`vitest run`**); **`test:watch`**, **`test:lib`**, **`test:components`**, **`test:hooks`**, **`test:changed`** slice it |
| `npm run ci`               | Lint + typecheck + Vitest + **`npm run build`** — the local gate, same steps as CI's **`test-web`** job                                      |
| `npm run ci:quick`         | Typecheck + **`test:lib`** + **`test:components`** — skips lint and the build                                                                |
| `npm run e2e` / `e2e:open` | Cypress E2E (headless / browser) against a running app; default baseUrl `http://localhost:5173`                                              |
| `npm run e2e:vite`         | `start-server-and-test` on **:5173** + headless Cypress with mocked FastAPI — complements Vitest, not a substitute                           |
| `npm run ci:e2e`           | **`npm run build`** then **`e2e:vite`** (full UI smoke with mocked backend)                                                                  |
| `npm run clean`            | Deletes **`dist-vite/`** (the only build output this app produces)                                                                           |

## Project Structure

From the monorepo root, this app lives at **`apps/web/`**:

```
apps/web/
├── app/                 # Route pages + layouts (`(app)/`, `(marketing)/`, `globals.css`)
├── components/          # Shared React components
├── features/            # Feature-local modules
├── hooks/               # `use*` hooks and Zustand stores
├── lib/                 # API client and helpers (e.g. `lib/api.ts`)
├── vite/                # `routes.ts` (page discovery) + `next-compat/` import shims
├── cypress/             # E2E specs
├── vite-entry.tsx       # SPA entry point
├── vite.config.ts
├── vitest.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

## Connecting to Backend

Set **`NEXT_PUBLIC_API_URL`** (see **`.env.example`**, copied to **`.env.local`** in dev) to your API base URL, usually `http://localhost:8000`.

Start the API from the repo root:

```bash
python3 apps/api/server/main.py
# or: cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000
```

### Troubleshooting (“web doesn’t work”)

1. **Node 20+** — match **`.nvmrc`**; run `node -v`.
2. **Install & build** — from `apps/web`: `npm ci` (or `npm install`), then `npm run dev` or `npm run ci` to match CI.
3. **Environment** — copy **`.env.example`** → **`.env.local`**. The live stack (vite on `:5173`, same-origin in production) authenticates via FastAPI **`/auth/login`** — it does not serve `/api/auth/*`. The **`NEXTAUTH_*`** values in the template are inert Next-runtime leftovers: nothing under Vite reads them, so they can stay untouched.
4. **API must be up** — the UI calls **`NEXT_PUBLIC_API_URL`** (default `http://localhost:8000`). Login and most actions use the FastAPI backend; if the API is down, the home page shows **offline** and login will fail.
5. **GitHub OAuth** — not available under Vite. `next-auth/react` resolves to the compatibility stub **`vite/next-compat/next-auth.tsx`**, whose `useSession` always reports `unauthenticated` and whose `signIn` is a no-op, so **`GITHUB_ID`** / **`GITHUB_SECRET`** are never consulted. Use the **`/login`** form (FastAPI `/auth/login`) for username/password. Vite's dev server serves only the `planner`/`calendar` handlers under `app/api`.

## Training console

The **Training** page calls `POST /training/start`. Native trainer `.soul` files on the API host embed `stoi` / `itos` / `chars` for fair `cli.py eval`; formats and caveats are in [docs/policies/CONTRIBUTING.md](../../docs/policies/CONTRIBUTING.md) (_Checkpoint vocabulary_).

## License

MIT
