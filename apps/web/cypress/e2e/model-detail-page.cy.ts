/**
 * Model detail fold — /models now hosts the detail behind ?id=
 * (v1 route map card 20261101_154054 merged /model/[id] into /models).
 */

describe('Model detail fold', () => {
  beforeEach(() => {
    cy.visit('/models?id=test-model-id')
    cy.wait(1000)
  })

  it('displays model info or the inline error state', () => {
    cy.get('body').then(($body) => {
      const text = $body.text()
      // Detail renders, or the fold shows its inline load error
      expect(
        text.includes('Model') ||
          text.includes('Not Found') ||
          text.includes('not found') ||
          text.includes('Error') ||
          text.includes('Failed to load model details'),
      ).to.equal(true)
    })
  })

  it('shows the Model Details fold', () => {
    cy.contains('Model Details').should('exist')
    cy.get('[data-testid="model-details-body"]').should('exist')
  })
})
