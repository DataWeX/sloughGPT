/**
 * Home page - dashboard overview
 */
describe('Home page - dashboard', () => {
  it('displays the dashboard title', () => {
    cy.visit('/')
    cy.wait(500)
    cy.contains('h1', 'SloughGPT').should('be.visible')
  })

  it('shows API status card', () => {
    cy.visit('/')
    cy.wait(500)
    cy.contains('API status').should('be.visible')
  })

  it('shows quick actions section', () => {
    cy.visit('/')
    cy.wait(500)
    cy.contains('Quick actions').should('be.visible')
  })
})
