/**
 * Phase 1 Vite smoke — boots shell and walks Sidebar nav routes.
 * Run: cypress run --config baseUrl=http://127.0.0.1:5173 --spec cypress/e2e/vite-phase0-smoke.cy.ts
 */

const NAV_PATHS = [
  '/settings',
  '/collections', // Phase 4: legacy → /datasets
  '/datasets',
  '/knowledge',
  '/personality',
  '/agents',
  '/souls',
  '/monitoring',
  '/developer',
  '/feedback',
  '/training',
  '/training/runs',
  '/profile',
  '/errors', // should redirect → /monitoring
]

// FastAPI may be down during shell smoke; ignore client rejections from http-client.
Cypress.on('uncaught:exception', (err) => {
  if (
    err.message.includes('Connection unavailable') ||
    err.message.includes('Failed to fetch') ||
    err.message.includes('NetworkError') ||
    err.message.includes('load failed')
  ) {
    return false
  }
  return true
})

function assertShell(path: string) {
  cy.visit(path, { failOnStatusCode: false })
  cy.get('#root', { timeout: 20000 }).should('exist')
  cy.get('vite-error-overlay', { timeout: 5000 }).should('not.exist')
  cy.get('body').should('not.contain', 'Application error')
  cy.get('body').should('not.contain', 'Something went wrong')
}

describe('vite shell phase1', () => {
  it('boots collections via legacy redirect', () => {
    assertShell('/collections')
    cy.location('pathname').should('eq', '/datasets')
  })

  NAV_PATHS.forEach((path) => {
    it(`renders shell at ${path}`, () => {
      assertShell(path)
      if (path === '/errors' || path === '/collections') {
        const expected = path === '/errors' ? '/monitoring' : '/datasets'
        cy.location('pathname', { timeout: 10000 }).should('eq', expected)
      } else {
        cy.location('pathname').should('eq', path)
      }
    })
  })

  it('sidebar home link navigates', () => {
    assertShell('/settings')
    cy.get('a[href="/"]').first().click({ force: true })
    cy.location('pathname', { timeout: 10000 }).should('not.eq', '/settings')
  })
})
