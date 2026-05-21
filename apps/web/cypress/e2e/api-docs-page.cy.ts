/**
 * API Docs page - endpoint documentation
 */
describe('API Docs page', () => {
  it('loads the API docs page', () => {
    cy.visit('/api-docs')
    cy.wait(500)
    cy.get('body').should('contain', 'API')
  })
})
