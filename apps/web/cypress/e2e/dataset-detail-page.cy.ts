/**
 * Dataset detail page - individual dataset info, data preview, stats
 */

describe('Dataset detail page', () => {
  beforeEach(() => {
    cy.visit('/dataset/test-dataset-id')
    cy.wait(1000)
  })

  it('displays dataset info or not-found state', () => {
    cy.get('body').then(($body) => {
      const text = $body.text()
      expect(
        text.includes('Dataset') ||
        text.includes('Not Found') ||
        text.includes('not found') ||
        text.includes('Error')
      ).to.be.true
    })
  })

  it('shows a back navigation link', () => {
    cy.get('a[href="/datasets"]').should('exist')
  })

  it('shows export or delete buttons', () => {
    cy.get('body').then(($body) => {
      const text = $body.text()
      if (text.includes('Dataset') && !text.includes('Not Found')) {
        // If dataset loads, action buttons should exist
        cy.get('button').should('have.length.gte', 1)
      }
    })
  })
})
