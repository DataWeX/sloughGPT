describe('Compare page', () => {
  it('renders and shows model table', () => {
    cy.on('uncaught:exception', () => false)
    cy.mockHealth({ model_loaded: true, model_type: 'gpt2' })
    cy.mockCompare()
    cy.visit('/compare', { timeout: 60000, failOnStatusCode: false })
    cy.contains('h1', 'Compare', { timeout: 30000 }).should('be.visible')
    cy.contains('th', 'Model', { timeout: 15000 }).should('be.visible')
    cy.contains('gpt2', { timeout: 15000 }).should('be.visible')
    cy.contains('Loaded', { timeout: 10000 }).should('be.visible')
    cy.contains('3 models available', { timeout: 10000 }).should('be.visible')
    cy.contains('button', 'Refresh', { timeout: 5000 }).should('be.visible')
  })
})
