describe('Agents page', () => {
  it('renders and shows agents list and create button', () => {
    cy.on('uncaught:exception', () => false)
    cy.mockAgents()
    cy.visit('/agents', { timeout: 60000, failOnStatusCode: false })
    cy.contains('Agents', { timeout: 30000 }).should('be.visible')
    cy.contains('button', 'New Agent', { timeout: 15000 }).should('be.visible')
  })
})
