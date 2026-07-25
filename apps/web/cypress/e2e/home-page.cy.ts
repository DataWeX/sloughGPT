describe('Home page - dashboard', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/')
  })

  it('displays the dashboard title', () => {
    cy.contains('h1', 'SloughGPT').should('be.visible')
  })

  it('shows API status card', () => {
    cy.contains('API status').should('be.visible')
  })

  it('shows quick actions with agent-forward labels', () => {
    cy.contains('Start chatting').should('be.visible')
    cy.contains('Personalities').should('be.visible')
    cy.contains('Teach me').should('be.visible')
    cy.contains('Datasets').should('be.visible')
  })

  it('links quick actions to correct routes', () => {
    cy.contains('a', 'Personalities').should('have.attr', 'href', '/models')
    cy.contains('a', 'Teach me').should('have.attr', 'href', '/training')
  })
})
