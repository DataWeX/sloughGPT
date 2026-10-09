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
    // Scope to main content: sidebar nav links (collapsed, not visible) also
    // contain these words and would match cy.contains first.
    cy.get('.sl-app-content').contains('Capabilities').should('be.visible')
    cy.get('.sl-app-content').contains('Training').scrollIntoView().should('be.visible')
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

describe('VQA flow', () => {
  beforeEach(() => {
    cy.on('uncaught:exception', () => false)
    cy.mockAll()
    cy.visit('/multimodal')
  })

  it.skip('shows VQA section — never built; core-feature backlog (features-to-core list)', () => {
    cy.contains('Visual Question Answering').scrollIntoView().should('be.visible')
  })

  it.skip(
    'accepts a question input — never built; core-feature backlog (features-to-core list)',
    () => {
      cy.get('input[aria-label="Question"]').scrollIntoView().type('What is in this image?')
      cy.get('input[aria-label="Question"]').should('have.value', 'What is in this image?')
    },
  )
})

describe('Object detection flow', () => {
  beforeEach(() => {
    cy.on('uncaught:exception', () => false)
    cy.mockAll()
    cy.visit('/multimodal')
  })

  it.skip('shows object detection section — never built; core-feature backlog', () => {
    cy.contains('Object Detection').scrollIntoView().should('be.visible')
  })
})
