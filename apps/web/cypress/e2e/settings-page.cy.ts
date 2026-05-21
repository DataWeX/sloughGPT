/**
 * Settings page - application configuration
 */
describe('Settings page', () => {
  it('loads the settings page', () => {
    cy.visit('/settings')
    cy.get('body').should('not.be.empty')
  })

  it('shows settings heading', () => {
    cy.visit('/settings')
    cy.wait(1000)
    cy.get('body').should('contain', 'Settings')
  })
})
