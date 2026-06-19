describe('Knowledge page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/knowledge')
  })

  it('renders the page header', () => {
    cy.contains('Knowledge').should('be.visible')
    cy.contains('Add Knowledge').should('be.visible')
    cy.contains('Knowledge Base').should('be.visible')
  })

  it('allows adding a knowledge item', () => {
    cy.mockKnowledge()
    cy.get('textarea').type('Test knowledge content')
    cy.contains('Add').click()
    cy.wait('@mockKnowledgeAdd')
    cy.contains('Test knowledge content').should('be.visible')
  })

  it('shows empty state when no items', () => {
    cy.contains('Add your first knowledge item above').should('be.visible')
  })

  it('search filters items', () => {
    cy.mockKnowledge(['item1', 'item2'])
    cy.get('input[placeholder*="Search"]').type('item1')
    cy.contains('item1').should('be.visible')
    cy.contains('item2').should('not.exist')
  })

  it('allows batch delete', () => {
    cy.mockKnowledge(['itemA', 'itemB'])
    cy.get('input[type="checkbox"]').first().check()
    cy.contains('Delete 1').should('be.visible').click()
    cy.wait('@mockKnowledgeDelete')
    cy.get('[data-testid="knowledge-list"]').children().should('have.length', 1)
  })
})
