describe('Teach me page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/training')
  })

  it('renders the page header', () => {
    cy.contains('Teach me').should('be.visible')
  })

  it('shows dataset selector', () => {
    cy.get('select, [role="combobox"]').should('exist')
  })

  it('shows start training button', () => {
    cy.contains('button', /start/i).should('exist')
  })

  it('accepts pasted text input', () => {
    cy.contains('button', /paste/i).click()
    cy.get('textarea').should('exist').type('This is test training data.')
    cy.get('textarea').should('have.value', 'This is test training data.')
  })

  it('shows checkpoints card', () => {
    cy.contains('Checkpoints').should('be.visible')
  })
})
