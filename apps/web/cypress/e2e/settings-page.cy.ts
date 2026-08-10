describe('Settings page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/settings')
  })

  it('renders the page header', () => {
    cy.contains('h1', 'Settings').should('be.visible')
  })

  it('shows appearance settings', () => {
    cy.contains('Appearance').should('be.visible')
  })

  it('shows chat defaults', () => {
    cy.contains('Chat defaults').scrollIntoView().should('be.visible')
  })

  it('has theme toggle', () => {
    cy.contains('button', 'Light').should('exist')
    cy.contains('button', 'Dark').should('exist')
    cy.contains('button', 'System').should('exist')
  })

  it('has temperature slider', () => {
    cy.get('input[type="range"]').should('exist')
  })
})
