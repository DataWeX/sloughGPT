/** Phase 1: next/navigation redirect() digest → react-router Navigate. */
/** Phase 4: legacy proxy.ts redirects (server 307 + client Navigate). */
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

describe('vite redirect digest', () => {
  it('/errors redirects to /monitoring', () => {
    cy.visit('/errors', { failOnStatusCode: false })
    cy.get('#root', { timeout: 15000 }).should('exist')
    cy.get('vite-error-overlay').should('not.exist')
    cy.location('pathname', { timeout: 15000 }).should('eq', '/monitoring')
  })

  it('page-less source /voice → /chat with query', () => {
    cy.visit('/voice', { failOnStatusCode: false })
    cy.get('#root', { timeout: 15000 }).should('exist')
    cy.location('pathname', { timeout: 15000 }).should('eq', '/chat')
    cy.location('search').should('eq', '?mode=talk')
  })

  it('shadowed source /infer → /models', () => {
    cy.visit('/infer', { failOnStatusCode: false })
    cy.get('#root', { timeout: 15000 }).should('exist')
    cy.location('pathname', { timeout: 15000 }).should('eq', '/models')
  })

  it('server middleware answers 307 for /collections', () => {
    cy.request({ url: '/collections', followRedirect: false, failOnStatusCode: false }).then(
      (res) => {
        expect(res.status).to.eq(307)
        expect(res.headers.location).to.eq('/datasets')
      },
    )
  })
})
