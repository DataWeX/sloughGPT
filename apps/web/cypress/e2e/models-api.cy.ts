/** Models page - basic load test. */

describe('Models page', () => {
  it('page loads without error', () => {
    cy.visit('/models')
    cy.get('body').should('not.be.empty')
    cy.wait(1000)
    cy.get('h1, h2').should('exist')
  })
})
