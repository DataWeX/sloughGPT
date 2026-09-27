describe('Agents page', () => {
  beforeEach(() => {
    cy.mockApiFallback()
    cy.mockHealth()
    cy.mockAgents()
    cy.visit('/agents')
    // Startup overlay (z-9999) can cover the card actions — hide it like the
    // chat/datasets specs do so clicks land on the real buttons.
    cy.window().then((win) => {
      win.document.head.insertAdjacentHTML(
        'beforeend',
        '<style>[class*="9999"]{display:none !important}</style>',
      )
    })
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
    cy.contains('button', 'Run').first().click()
    cy.get('input[placeholder*="What should"]').type('Write a poem{enter}')
    cy.wait('@agentsExecute')
    cy.contains('simulated agent response').should('exist')
  })

  it('deletes an agent', () => {
    cy.get('[aria-label^="Delete "]').first().click()
    cy.get('[role="alertdialog"]').contains('button', 'Delete').click()
    cy.wait('@agentsDelete')
  })
})
