/**
 * Chat page - basic load test
 */
describe('Chat page', () => {
  it('loads the chat page', () => {
    cy.visit('/chat')
    cy.wait(500)
    cy.get('body').should('not.be.empty')
  })
})
