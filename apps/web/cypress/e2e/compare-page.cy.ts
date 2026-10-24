/**
 * Compare journey — side-by-side model comparison.
 *
 * The legacy /compare page is redirect-shadowed (→ /benchmark): the compare
 * components moved into the benchmark page's "Compare" tab. This spec follows
 * the journey to its live home.
 */

describe('Compare (benchmark page)', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/benchmark')
  })

  it('displays the benchmark page title', () => {
    cy.contains('h1', 'Benchmark').should('be.visible')
  })

  it('shows the compare empty state', () => {
    cy.contains('button', 'Compare').click()
    cy.get('.sl-app-content').within(() => {
      cy.contains('No models available').should('be.visible')
      cy.contains('Select models above and click Run to compare them side by side.').should(
        'be.visible',
      )
    })
  })

  it('shows the Run compare button', () => {
    cy.contains('button', 'Compare').click()
    cy.get('.sl-app-content').within(() => {
      cy.contains('button', /Run on \d+ model/).should('exist')
    })
  })
})
