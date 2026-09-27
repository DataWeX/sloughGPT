/**
 * Single source of truth for the API origin used by e2e intercepts.
 *
 * The client resolves `NEXT_PUBLIC_API_URL` at build time (vite.config.ts
 * `inlinePublicEnv`), so mocks must target the same origin — otherwise every
 * request bypasses cy.intercept() and dies against a dead port.
 */
const fromConfig = Cypress.env('apiUrl')

export const apiBase: string =
  typeof fromConfig === 'string' && fromConfig.length > 0 ? fromConfig : 'http://localhost:8000'
