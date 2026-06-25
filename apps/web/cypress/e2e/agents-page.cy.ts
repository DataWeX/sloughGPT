describe('Agents page', () => {
  beforeEach(() => {
    cy.mockHealth()
    cy.mockAgents()
    cy.visit('/agents')
  })

  it('renders the page with agent list', () => {
    cy.contains('Agents').should('exist')
    cy.contains('Assistant').should('exist')
    cy.contains('Coder').should('exist')
  })

  it('shows create agent form', () => {
    cy.contains('New Agent').should('exist')
  })

  it('shows stats', () => {
    cy.contains('Total Agents').should('exist')
    cy.contains('Tool Assignments').should('exist')
    cy.contains('Available Tools').should('exist')
  })

  it('executes an agent inline', () => {
    cy.contains('Run').first().click()
    cy.get('input[placeholder*="What should"]').type('Write a poem{enter}')
    cy.wait('@agentsExecute')
    cy.contains('simulated agent response').should('exist')
  })

  it('deletes an agent', () => {
    cy.get('.text-destructive-foreground').first().click()
    cy.wait('@agentsDelete')
  })
})
