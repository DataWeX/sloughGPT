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

  it('has theme picker tiles', () => {
    cy.contains('button', 'Light').should('exist')
    cy.contains('button', 'Dark').should('exist')
    cy.contains('button', 'High Contrast').should('exist')
  })

  it('marks the applied canvas selected', () => {
    // Scope to the Theme group: the page also carries "Color theme" and
    // "Palette" radiogroups (present whenever the nav dialog is mounted),
    // each with its own checked radio — a global count is the wrong invariant.
    cy.get('[role="radiogroup"][aria-label="Theme"] [role="radio"][aria-checked="true"]').should(
      'have.length',
      1,
    )
    cy.contains('button', 'Dark').should('have.attr', 'aria-checked', 'true')
  })

  it('applies High Contrast to <html> when its tile is picked', () => {
    cy.contains('button', 'High Contrast').click()
    // high-contrast layers ON the dark canvas — both classes must land.
    cy.document().its('documentElement').should('have.class', 'high-contrast')
    cy.document().its('documentElement').should('have.class', 'dark')
    cy.document().its('documentElement').should('not.have.class', 'light')
    cy.contains('button', 'High Contrast').should('have.attr', 'aria-checked', 'true')
    // Persists across a reload (bootstrap reads it back before React mounts).
    cy.reload()
    cy.document().its('documentElement').should('have.class', 'high-contrast')
  })

  it('returns to light via Reset', () => {
    cy.contains('button', 'High Contrast').click()
    cy.contains('button', 'Reset').click()
    cy.document().its('documentElement').should('have.class', 'light')
    cy.document().its('documentElement').should('not.have.class', 'high-contrast')
  })

  it('has temperature slider', () => {
    cy.get('input[type="range"]').should('exist')
  })
})
