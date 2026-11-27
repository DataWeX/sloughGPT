/**
 * The API-vs-navigation kind-rule, declared once and shared by the support
 * mocks and any spec that registers its own GET intercept.
 *
 * Kept separate from api-mocks.ts on purpose: that file has module-level side
 * effects (Cypress.Commands.add), so importing it from a spec would re-register
 * every command. This module is pure.
 */

/**
 * Kind-rule: decide by WHAT the request is, not where it points.
 *
 * With apiBase '' (same-origin, card 20261123_135845) an API intercept pattern
 * is root-relative, so a path that shares its name with a page route — /agents,
 * /datasets, /models, /knowledge — matches BOTH the app's fetch()/XHR calls AND
 * cy.visit()'s document navigation. Replying JSON to the navigation fails
 * cy.visit with `content-type: application/json` (responses must be
 * text/html): 27 such failures were invisible while Cypress.on('fail') swallowed
 * every failure (card 20261125_130530).
 *
 * Before the same-origin switch apiBase was http://localhost:8000, so these
 * patterns could never match a visit to the dev origin — the collision is new.
 *
 * sec-fetch-dest is the browser's own classification: 'empty' for fetch()/XHR
 * (the only true API traffic), 'document'/'script'/'style'/'image'/'font' for
 * everything else. The Accept check is belt-and-braces — the same kind-rule
 * vite/api-relay-plugin.ts and the gateway use.
 */
export const isApiRequest = (req: { headers: Record<string, unknown> }): boolean => {
  const dest = req.headers['sec-fetch-dest']
  if (typeof dest === 'string' && dest !== '' && dest !== 'empty') return false
  const accept = req.headers['accept']
  if (typeof accept === 'string' && accept.includes('text/html')) return false
  return true
}

/**
 * Register a GET mock that answers API traffic only. A document navigation to
 * the same path is continued to the dev server, which serves the SPA shell —
 * letting a page route and an API path share a name safely.
 */
export const apiGet = (url: string, statusCode: number, body: unknown) =>
  cy.intercept('GET', url, (req) => {
    if (isApiRequest(req)) {
      req.reply({ statusCode, body })
    } else {
      req.continue()
    }
  })
