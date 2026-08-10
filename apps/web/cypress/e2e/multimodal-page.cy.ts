describe('Multimodal page', () => {
  beforeEach(() => {
    cy.on('uncaught:exception', () => false)
    cy.mockAll()
    cy.visit('/multimodal')
  })

  it('renders the page header', () => {
    cy.contains('h1', 'Multimodal').should('be.visible')
  })

  it('shows capability and training cards', () => {
    cy.contains('Capabilities').should('be.visible')
    cy.contains('Training').scrollIntoView().should('be.visible')
  })

  it('shows image training and batch training cards', () => {
    cy.contains('Image Training').scrollIntoView().should('be.visible')
    cy.contains('Train with multiple images').scrollIntoView().should('be.visible')
  })

  it('shows dataset, DPO, generation, and audio cards', () => {
    cy.contains('Image description dataset').scrollIntoView().should('be.visible')
    cy.contains('DPO fine-tune').scrollIntoView().should('be.visible')
    cy.contains('Image Generation').scrollIntoView().should('be.visible')
    cy.contains('Audio').scrollIntoView().should('be.visible')
  })

  it('accepts an image generation prompt', () => {
    cy.get('input[aria-label="Image generation prompt"]').scrollIntoView().type('A cat in a spacesuit')
    cy.get('input[aria-label="Image generation prompt"]').should('have.value', 'A cat in a spacesuit')
  })
})
