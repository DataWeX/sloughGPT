describe('Labs page', () => {
  beforeEach(() => {
    cy.mockHealth({ model_loaded: false, model_type: null })
    cy.visit('/labs')
  })

  it('renders the page title', () => {
    cy.contains('Playground').should('be.visible')
  })

  it('shows chat and train tabs', () => {
    cy.contains('button', 'Chat').should('be.visible')
    cy.contains('button', 'Train').should('be.visible')
  })

  it('shows model status card', () => {
    cy.contains('Model Status').should('be.visible')
  })

  it('shows quick train card when no model loaded', () => {
    cy.contains('Quick Train').should('be.visible')
  })

  it('shows training text input', () => {
    cy.get('textarea[placeholder*="training"]').should('exist')
  })

  it('shows vision fold section', () => {
    cy.contains('Vision', { matchCase: false }).should('be.visible')
  })
})
