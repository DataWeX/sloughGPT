/**
 * Training page - job management
 */
describe('Training page', () => {
  it('loads the training page', () => {
    cy.visit('/training')
    cy.get('body').should('not.be.empty')
    cy.wait(1000)
    cy.get('body').should('contain', 'Training')
  })
})
