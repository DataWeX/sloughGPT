describe('Export page', () => {
  it('renders and shows all sections', () => {
    cy.on('uncaught:exception', () => false)
    cy.mockHealth({ model_loaded: true, model_type: 'gpt2' })
    cy.mockExport()
    cy.visit('/export', { timeout: 60000, failOnStatusCode: false })
    cy.contains('Export', { timeout: 30000 }).should('be.visible')
    cy.contains('Model Export', { timeout: 15000 }).should('be.visible')
    cy.contains('button', 'SOU', { timeout: 10000 }).should('be.visible')
    cy.contains('Training Data Export', { timeout: 5000 }).should('be.visible')
    cy.get('input[placeholder="Output path"]', { timeout: 10000 }).should('be.visible')
    cy.contains('button', 'Export', { timeout: 5000 }).should('be.enabled')
  })
})
