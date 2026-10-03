# SloughGPT Web UI

Modern TypeScript-based web interface for SloughGPT using Base UI components.

## Features

- 💬 **Chat Interface** - Real-time conversations with AI models
- 📊 **Dataset Management** - Create, view, and manage training datasets
- 🤖 **Model Management** - Browse and configure available AI models
- 📈 **Monitoring Dashboard** - Real-time system metrics and health checks
- 🔄 **Real-time Updates** - Live data with WebSocket support
- 📱 **Progressive Web App** - Installable with offline capabilities

## Tech Stack

- **Framework**: Vite + React 19, TypeScript, React Router v7
- **UI Library**: Base UI
- **State Management**: Zustand
- **Charts**: Recharts
- **Styling**: Tailwind CSS

## Getting Started

### Prerequisites

- **Node.js 20** (match repo root **`.nvmrc`**; CI uses the same major)
- npm or yarn

### Installation

```bash
# From repository root — frontend lives under apps/web
cd apps/web

# Install dependencies
npm install

# Or with yarn
yarn install
```

### Development

```bash
# Start development server (Vite)
npm run dev

# The app will open at http://localhost:5173
```

Before pushing changes, run the same checks as CI: **`npm ci && npm run ci`** (from this directory — lint, typecheck, Vitest, `vite build`; same as CI **`frontend-build`**).

**Talking to models:** set **`NEXT_PUBLIC_API_URL`** to your FastAPI base (default `http://localhost:8000`). Use **Models** to **`POST /models/load`** (`model_id` in JSON), then **Chat** sends message history to **`POST /chat/stream`** (SSE) with fallback to **`POST /chat`**. Older single-prompt paths **`/inference/generate/stream`** / **`/inference/generate`** remain available for other clients. The client **`modelController.load()`** matches that contract (not `/models/{id}/load`).

**UI vs core engine:** this app is **only** HTML/CSS/TS and `fetch` to the API. It does **not** bundle or import Python (`packages/core-py`, trainers, or inference kernels). Deploy the Vite build on any static/hosted frontend; run the FastAPI process separately — the only link is **`NEXT_PUBLIC_API_URL`** and the JSON contracts in **`lib/http-client.ts`** — the single source of truth for every API call, never bypass it with raw `fetch` (see also **`docs/STRUCTURE.md`**).

**Cypress E2E** (mocked API, no Python process): after a production build (`npm run build`), run **`npm run e2e:vite`** (starts `dev:vite` on port **5173**). Or `npm run dev` and **`npm run e2e`** / **`npm run e2e:open`**.

### Build for Production

```bash
npm run build        # vite build → dist-vite/
npm run preview      # serve the production build locally
```

### Docker image

The **`file:../../packages/strui`** dependency must resolve, so build the image from the **repository root**:

```bash
# Vite static SPA (compose `web` service, port 3000)
docker build -f apps/web/Dockerfile.vite -t sloughgpt/web:latest .
docker compose -f infra/docker/docker-compose.yml up -d web
```

**`infra/docker/docker-compose.yml`** `web` service uses `context: ../..` and `dockerfile: apps/web/Dockerfile.vite`. Root **`.dockerignore`** excludes **`apps/web/.next`**, **`apps/web/dist-vite`**, and **`**/node_modules`** (source stays in context for image builds).

## API Configuration

The web UI connects to the FastAPI backend. By default, it expects the API at `http://localhost:8000`.

To change the API URL, copy **`.env.example`** to **`.env.local`** (or edit **`.env`**) and set **`NEXT_PUBLIC_API_URL`** — see **`lib/config.ts`** (`http://localhost:8000` by default).

## Available Scripts

| Script                     | Description                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `npm run dev`              | Start Vite dev server (`http://localhost:5173`)                                      |
| `npm run build`            | Vite production build → **`dist-vite/`**                                             |
| `npm run preview`          | Serve the production build locally                                                   |
| `npm run lint`             | Run ESLint                                                                           |
| `npm run typecheck`        | TypeScript `tsc --noEmit`                                                            |
| `npm run test`             | Vitest unit tests                                                                    |
| `npm run ci`               | Lint + typecheck + Vitest + **`build:vite`** (parity with CI **`frontend-build`**)   |
| `npm run ci:vite`          | Typecheck + Vitest + **`build:vite`**                                                |
| `npm run e2e` / `e2e:open` | Cypress E2E (browser) against a running app; default baseUrl `http://localhost:5173` |
| `npm run e2e:vite`         | Cypress Vite smoke + redirect specs against `dev:vite` (CI **`frontend-e2e`**)       |

## Project Structure

From the monorepo root, this app lives at **`apps/web/`**:

```
apps/web/
├── app/                 # Route page components (loaded by vite/routes.ts)
├── components/          # Shared React components
├── lib/                 # API client and helpers (e.g. `lib/http-client.ts`)
├── vite/                # Vite plugins, next-compat shims, route helpers
├── vite-entry.tsx       # SPA shell + React Router routes
├── vite.config.ts
├── index.html
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
3. **Environment** — copy **`.env.example`** → **`.env.local`**. Set **`NEXTAUTH_SECRET`** (e.g. `openssl rand -base64 32`) so auth can issue sessions. **`NEXTAUTH_URL`** should match where you open the app (e.g. `http://localhost:5173`).
4. **API must be up** — the UI calls **`NEXT_PUBLIC_API_URL`** (default `http://localhost:8000`). Login and most actions use the FastAPI backend; if the API is down, the home page shows **offline** and login will fail.
5. **GitHub OAuth** — optional. If **`GITHUB_ID`** / **`GITHUB_SECRET`** are unset, NextAuth still runs with a placeholder provider; use the **`/login`** form (FastAPI `/auth/login`) for username/password.

## Training console

The **Training** page calls `POST /training/start`. Native trainer `.soul` files on the API host embed `stoi` / `itos` / `chars` for fair `cli.py eval`; formats and caveats are in [docs/policies/CONTRIBUTING.md](../../docs/policies/CONTRIBUTING.md) (_Checkpoint vocabulary_).

## License

MIT
