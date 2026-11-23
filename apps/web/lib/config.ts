/** FastAPI backend base URL for the app (client bundles).
 *
 * Same-origin by contract (card 20261123_135845): the empty base makes
 * every request root-relative — the SERVER owns the port, the browser owns
 * exactly one origin. Dev relays through vite (`vite/api-relay-plugin`,
 * the gateway's kind-rule); the static build is served by the gateway
 * (MAN_STATIC_DIR), which byte-relays to Python. No `process.env` reaches
 * the browser: user `define` keys do not apply under vite 8 dev, so the
 * old expression was served raw and threw at import — killing
 * http-client's whole chain while vitest stayed green in Node. */
export const PUBLIC_API_URL = ''
