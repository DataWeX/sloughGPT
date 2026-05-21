describe('Tokenizer page', () => {
  it('renders and shows all sections', () => {
    cy.on('uncaught:exception', () => false)
    cy.mockTokenizer()
    cy.visit('/tokenizer', { timeout: 60000, failOnStatusCode: false })
    cy.contains('Tokenizer', { timeout: 30000 }).should('be.visible')
    cy.contains('50257', { timeout: 15000 }).should('be.visible')
    cy.contains('Samples', { timeout: 10000 }).should('be.visible')
    cy.contains('the', { timeout: 10000 }).should('be.visible')
    cy.contains('button', 'Playground', { timeout: 5000 }).click()
    cy.contains('button', 'Tokenize', { timeout: 5000 }).should('be.visible')
  })
})
