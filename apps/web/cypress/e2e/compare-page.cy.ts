/**
 * Compare page - side-by-side model comparison
 */

describe('Compare page', () => {
  beforeEach(() => {
    cy.visit('/compare')
    cy.wait(1000)
  })

  it('displays the compare page title', () => {
    cy.contains('h1', 'Compare').should('be.visible')
  })

  it('shows model comparison table or empty state', () => {
    cy.get('body').then(($body) => {
      if ($body.text().includes('No models')) {
        cy.contains('No models').should('be.visible')
      } else {
        cy.get('table').should('exist')
      }
    })
  })

  it('shows Refresh button', () => {
    cy.contains('button', /Refresh/).should('exist')
  })
})
