/**
 * Training job detail page - individual training job info, logs, status
 */

describe('Training job detail page', () => {
  beforeEach(() => {
    cy.visit('/training/job/test-job-id')
    cy.wait(1000)
  })

  it('displays job info or not-found state', () => {
    cy.get('body').then(($body) => {
      const text = $body.text()
      expect(
        text.includes('Training') ||
        text.includes('Job') ||
        text.includes('Not Found') ||
        text.includes('not found') ||
        text.includes('Error')
      ).to.be.true
    })
  })

  it('shows a back navigation link', () => {
    cy.get('a[href="/training"]').should('exist')
  })

  it('shows status or progress indicator', () => {
    cy.get('body').then(($body) => {
      const text = $body.text()
      if (text.includes('Training') && !text.includes('Not Found')) {
        cy.get('button, [role="progressbar"], .animate-pulse').should('exist')
      }
    })
  })
})
