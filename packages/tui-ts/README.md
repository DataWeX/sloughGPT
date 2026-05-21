# @sloughgpt/tui

Terminal UI for SloughGPT using **[Ink](https://github.com/vadimdemedes/ink)** and the **[TypeScript SDK](../sdk-ts/typescript-sdk)**. Same HTTP contracts as `apps/web` and the Python `apps/tui` probes.

## Prerequisite

Build the SDK once (this package links to it via `file:`):

```bash
cd packages/sdk-ts/typescript-sdk && npm install && npm run build
```

## Install & run

```bash
cd packages/tui-ts
npm install
npm run build
npm start
# or: npx sloughgpt-tui   (after npm link / global install from this package)
```

- **API URL:** `SLOUGHGPT_API_URL` or `--url http://127.0.0.1:8000` (default port 8000).
- **Keys:** set `X-API-Key` via SDK when we add env support (same as web).

## Keys

| Key | Action |
|-----|--------|
| `r` | Refresh health |
| `q` / `Esc` | Quit |

## Scripts

| Script | Purpose |
|--------|---------|
| `npm run dev` | `tsc` + run CLI |
| `npm run ci` | lint + test + build |

## Roadmap

See [docs/plans/tui-cli-port.md](../../docs/plans/tui-cli-port.md) — more screens (train, export) and shared layout.
