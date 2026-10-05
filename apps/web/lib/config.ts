/** FastAPI backend base URL for the app (client bundles).
 *
 * `??` (not `||`) so an explicit empty value survives: `npm run build:vite`
 * builds same-origin, where the gateway relays the API — a `||` fallback to
 * `http://localhost:8000` there would reintroduce a second origin (CORS) for
 * every request the static site makes. */
export const PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
