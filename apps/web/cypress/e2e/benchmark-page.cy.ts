/**
 * Benchmark page - model performance comparison
 */
describe('Benchmark page', () => {
  it('loads the benchmark page', () => {
    cy.visit('/benchmark')
    cy.wait(500)
    cy.get('body').should('contain', 'Benchmark')
  })
})
