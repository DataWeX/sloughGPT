describe('Teach me page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.intercept('GET', 'http://localhost:8000/datasets', {
      statusCode: 200,
      body: {
        datasets: [
          { id: 'ds1', name: 'shakespeare', source: 'local', size: 1 },
          { id: 'ds2', name: 'tinyshakespeare', source: 'local', size: 1 },
        ],
      },
    }).as('datasetsList')
    cy.visit('/training')
  })

  it('renders the page header', () => {
    cy.contains('h1', 'Teach me').should('be.visible')
  })

  it('shows dataset selector', () => {
    cy.get('button[aria-label="Dataset selector"]').should('exist')
  })

  it('shows start training button', () => {
    cy.contains('button', /start/i).should('exist')
  })

  it('accepts pasted text input', () => {
    cy.contains('button', 'Show advanced settings').click()
    cy.contains('button', /^Paste text$/).click()
    cy.get('textarea[aria-label="Training text input"]').should('exist').type('This is test training data.')
    cy.get('textarea[aria-label="Training text input"]').should('have.value', 'This is test training data.')
  })

  it('shows the trained models card on the history tab', () => {
    cy.contains('button', 'History').click()
    cy.contains('Trained models').scrollIntoView().should('be.visible')
  })
})
