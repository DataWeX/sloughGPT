describe('Not found page', () => {
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
