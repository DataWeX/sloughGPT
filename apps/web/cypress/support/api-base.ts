/**
 * Single source of truth for the API origin used by e2e intercepts.
 *
 * Same-origin by contract (card 20261123_135845): the client sends
 * root-relative requests (PUBLIC_API_URL === ''), so intercepts must be
 * root-relative too — an absolute http://localhost:8000 base would never
 * match and every request would bypass cy.intercept(). Override via
 * `cypress run --env apiUrl=...` only when driving against a real backend.
 */
const fromConfig = Cypress.env('apiUrl')

export const apiBase: string =
  typeof fromConfig === 'string' && fromConfig.length > 0 ? fromConfig : ''
