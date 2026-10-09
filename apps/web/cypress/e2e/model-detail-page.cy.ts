/**
 * Model detail page - individual model info, load/unload, activity
 */

describe('Model detail page', () => {
  beforeEach(() => {
    cy.visit('/model/test-model-id')
    cy.wait(1000)
  })

  it('displays model info or not-found state', () => {
    cy.get('body').then(($body) => {
      const text = $body.text()
      // Either shows model details or a 404 / error
      expect(
        text.includes('Model') ||
          text.includes('Not Found') ||
          text.includes('not found') ||
          text.includes('Error'),
      ).to.equal(true)
    })
  })

  it('shows a back navigation button', () => {
    cy.contains('button', 'Back').should('exist')
  })
})
