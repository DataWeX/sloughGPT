describe('Home page - dashboard', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/')
  })

  it('displays the dashboard greeting header', () => {
    cy.get('.sl-app-content').within(() => {
      cy.contains(/(Good morning|Good afternoon|Good evening|Burning the midnight oil|Late night vibes)/).should('be.visible')
      cy.contains('gpt2 loaded').should('be.visible')
    })
  })

  it('shows the API online status', () => {
    cy.get('.sl-app-content').within(() => {
      cy.contains('Online').should('be.visible')
    })
  })

  it('shows quick actions with agent-forward labels', () => {
    cy.get('.sl-app-content').within(() => {
      cy.contains('Start chatting').should('be.visible')
      cy.contains('Personalities').should('be.visible')
      cy.contains('Teach me').should('be.visible')
      cy.contains('Datasets').should('be.visible')
    })
  })

  it('links quick actions to correct routes', () => {
    cy.get('.sl-app-content').within(() => {
      cy.contains('a[href="/chat"]', 'Start chatting').should('have.attr', 'href', '/chat')
      cy.contains('a[href="/models"]', 'Personalities').should('have.attr', 'href', '/models')
      cy.contains('a[href="/training"]', 'Teach me').should('have.attr', 'href', '/training')
      cy.contains('a[href="/datasets"]', 'Datasets').should('have.attr', 'href', '/datasets')
    })
  })
})
