/**
 * Datasets page - import, combine, delete datasets
 */

describe('Datasets page', () => {
  beforeEach(() => {
    cy.visit('/datasets')
    cy.wait(2000)
  })

  it('displays the datasets page', () => {
    cy.contains('Datasets').should('be.visible')
  })

  it('shows Import Dataset button', () => {
    cy.contains('button', 'Import Dataset').should('be.visible')
  })

  it('shows Combine button', () => {
    cy.contains('button', /Combine \(/).should('be.visible')
  })

  it('opens Import Dataset modal', () => {
    cy.contains('button', 'Import Dataset').click()
    cy.wait(200)
    cy.contains('Import Dataset').should('be.visible')
    cy.contains('GitHub').should('be.visible')
  })

  it('shows Refresh button', () => {
    cy.contains('button', 'Refresh').should('be.visible')
  })

  it('shows CLI commands section', () => {
    cy.contains('CLI').should('be.visible')
  })
})

describe('Dataset Import Modal', () => {
  beforeEach(() => {
    cy.visit('/datasets')
    cy.wait(500)
    cy.contains('button', 'Import Dataset').click()
    cy.wait(300)
  })

  it('defaults to GitHub source', () => {
    cy.contains('GitHub Repository URL').should('be.visible')
    cy.contains('Search Repos').should('be.visible')
  })

  it('switches to HuggingFace source', () => {
    cy.contains('button', 'HuggingFace').click()
    cy.contains('HuggingFace Dataset ID').should('be.visible')
  })

  it('switches to URL source', () => {
    cy.contains('button', 'URL').click()
    cy.contains('URL').should('be.visible')
  })

  it('switches to Local source', () => {
    cy.contains('button', 'Local').click()
    cy.contains('Path to file or directory').should('be.visible')
  })

  it('shows file type toggles for GitHub source', () => {
    cy.contains('.py').should('be.visible')
    cy.contains('.js').should('be.visible')
    cy.contains('.ts').should('be.visible')
  })

  it('shows dataset name input', () => {
    cy.contains('Dataset Name').should('be.visible')
  })
})

describe('Dataset Export', () => {
  it('shows Export button on dataset cards when datasets exist', () => {
    cy.visit('/datasets')
    cy.wait(2000)
    cy.get('body').then(($body) => {
      if ($body.text().includes('No datasets found')) {
        cy.contains('No datasets found').should('be.visible')
      } else {
        cy.contains('button', 'Export').should('exist')
      }
    })
  })
})
