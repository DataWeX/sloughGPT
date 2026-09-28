/**
 * Multimodal page — /multimodal is a legacy path that 307s to /models, which
 * then client-redirects to /developer (app/(app)/models/page.tsx). The
 * multimodal UI itself is unreachable via URL; its content is covered by
 * page.test.tsx (16 unit tests).
 */

describe('Multimodal page', () => {
  beforeEach(() => {
    cy.on('uncaught:exception', () => false)
    cy.mockAll()
    cy.visit('/multimodal')
  })

  it('redirects the legacy /multimodal path', () => {
    cy.url().should('include', '/models')
  })

  it('lands on the developer page after the client-side replace', () => {
    cy.url({ timeout: 20000 }).should('include', '/developer')
    // Header h1 sits inside an overflow container — clip breaks visibility,
    // so assert existence (pattern used by the other page specs).
    cy.contains('h1', 'Developer').should('exist')
  })
})
