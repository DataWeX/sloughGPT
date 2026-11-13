/**
 * Monitoring page - system metrics
 */
describe('Monitoring page', () => {
  it('loads the monitoring page', () => {
    cy.visit('/monitoring')
    cy.get('body').should('not.be.empty')
    cy.contains('System Health').should('be.visible')
  })
})
