/**
 * Compare page - side-by-side model comparison.
 *
 * /compare is a legacy path that 307s to /benchmark (lib/redirects.ts); the
 * comparison UI now lives in the benchmark page's Compare tab.
 */

describe('Compare page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/compare')
  })

  it('redirects to the benchmark page', () => {
    cy.url().should('include', '/benchmark')
    cy.contains('h1', 'Benchmark').should('be.visible')
  })

  it('shows the Compare tab with empty model selection', () => {
    cy.contains('button', 'Compare').click()
    cy.get('.sl-app-content').within(() => {
      cy.contains('Compare Models').should('be.visible')
      cy.contains('No models available').should('be.visible')
      cy.contains('Run benchmarks on multiple models to compare.').should('be.visible')
    })
  })

  it('disables Run until a model is picked', () => {
    cy.contains('button', 'Compare').click()
    cy.get('.sl-app-content').within(() => {
      cy.contains('button', 'Run on 0 models').should('be.disabled')
    })
  })
})
