describe('Vision page', () => {
  beforeEach(() => {
    cy.on('uncaught:exception', () => false)
    cy.mockHealth({ model_loaded: true })
    cy.mockVisual()
    cy.mockMultimodal()
    cy.visit('/vision', { timeout: 60000 })
  })

  it('renders header, tabs, analyze content', () => {
    cy.contains('h1', 'Vision Studio', { timeout: 30000 }).should('be.visible')
    cy.contains('Analyze', { timeout: 20000 }).should('be.visible')
    cy.contains('Supervised Train').should('be.visible')
    cy.contains('Generate', { timeout: 20000 }).should('be.visible')
    cy.contains('History').should('be.visible')
    cy.get('input[placeholder*="question"]', { timeout: 20000 }).should('be.visible')
    cy.contains('Drop an image here').should('be.visible')
  })

  it('checkpoints, Load/Delete, History tab', () => {
    cy.wait('@visualCheckpoints', { timeout: 20000 }).its('response.statusCode').should('eq', 200)
    cy.contains('Visual Checkpoints', { timeout: 15000 }).should('be.visible')
    cy.contains('vision-v2').should('be.visible')
    cy.contains('vision-v1').should('be.visible')
    cy.contains('Qwen2.5-0.5B-Instruct').should('be.visible')
    cy.contains('88.3%').should('be.visible')
    cy.contains('vision-v2').parents('[class*="flex items-center justify-between"]').within(() => {
      cy.contains('button', 'Load').click({ force: true })
    })
    cy.wait('@visualLoad', { timeout: 15000 }).its('response.statusCode').should('eq', 200)
    cy.contains('vision-v1').parents('[class*="flex items-center justify-between"]').first().within(() => {
      cy.get('button[class*="text-destructive"]').click({ force: true })
    })
    cy.wait('@visualDelete', { timeout: 15000 }).its('response.statusCode').should('eq', 200)
    cy.wait('@multimodalReport', { timeout: 20000 })
    cy.contains('History', { timeout: 20000 }).click()
    cy.contains('Images learned', { timeout: 10000 }).should('be.visible')
    cy.contains('Vocab size').should('be.visible')
    cy.contains('Mean accuracy').should('be.visible')
    cy.contains('64.2%').should('be.visible')
  })
})
