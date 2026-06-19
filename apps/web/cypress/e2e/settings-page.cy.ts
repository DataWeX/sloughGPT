describe('Settings page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/settings')
  })

  it('renders the page header', () => {
    cy.contains('Settings').should('be.visible')
  })

  it('shows appearance settings', () => {
    cy.contains('Appearance').should('be.visible')
  })

  it('shows chat defaults', () => {
    cy.contains('Chat defaults').should('be.visible')
  })

  it('has theme toggle', () => {
    cy.get('[role="combobox"], select').should('exist')
  })

  it('has temperature slider', () => {
    cy.get('input[type="range"]').should('exist')
  })
})
