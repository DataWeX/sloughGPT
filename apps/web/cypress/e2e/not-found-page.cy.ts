describe('Not found page', () => {
  beforeEach(() => {
    // Hermetic: without intercepts the client hits the dead API origin and the
    // startup overlay (z-9999) stays up, hiding the 404 content entirely.
    cy.mockApiFallback()
    cy.mockHealth()
    cy.mockModels()
  })

  it('renders 404 page for unknown routes', () => {
    cy.on('uncaught:exception', () => false)
    cy.visit('/nonexistent-route-test', { failOnStatusCode: false, timeout: 30000 })
    cy.get('h1', { timeout: 20000 }).should('contain', 'not found')
  })

  it('shows navigation buttons', () => {
    cy.on('uncaught:exception', () => false)
    cy.visit('/nonexistent-route-test', { failOnStatusCode: false, timeout: 30000 })
    cy.contains('a', 'Home', { timeout: 20000 }).should('be.visible')
  })
})
