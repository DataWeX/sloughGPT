/**
 * Compare page - side-by-side model comparison
 */

describe('Compare page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/compare')
  })

  it('displays the compare page title', () => {
    cy.contains('h1', 'Model Comparison').should('be.visible')
  })

  it('shows model comparison table or empty state', () => {
    cy.get('.sl-app-content').within(() => {
      cy.contains('No models available').should('be.visible')
      cy.contains('No benchmark results yet.').should('be.visible')
    })
  })

  it('shows Refresh button', () => {
    cy.get('.sl-app-content').within(() => {
      cy.contains('button', /Benchmark all/).should('exist')
    })
  })
})
