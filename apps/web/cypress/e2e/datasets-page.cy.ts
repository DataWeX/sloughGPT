/**
 * Datasets page - import, list, export, delete datasets
 */

const API = 'http://localhost:8000'

describe('Datasets page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/datasets')
  })

  it('renders the page header', () => {
    cy.contains('h1', 'Datasets').should('be.visible')
  })

  it('shows Import button', () => {
    cy.contains('button', /^Import$/).should('be.visible')
  })

  it('shows Refresh button', () => {
    cy.contains('button', 'Refresh').should('be.visible')
  })

  it('shows empty state when no datasets', () => {
    cy.contains('No datasets yet').should('be.visible')
    cy.contains('button', 'Import Dataset').should('be.visible')
  })

  it('opens the Import Dataset modal', () => {
    cy.contains('button', /^Import$/).click()
    cy.contains('Import Dataset').should('be.visible')
    cy.get('input[placeholder="Dataset name (optional)"]').should('exist')
  })
})

describe('Dataset Import Modal', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/datasets')
    cy.contains('button', /^Import$/).click()
  })

  it('defaults to Local Path source', () => {
    cy.get('[role="dialog"]').contains('button', 'Local Path').should('be.visible')
    cy.get('input[placeholder="/path/to/dataset/folder"]').should('be.visible')
  })

  it('shows all source tabs', () => {
    cy.get('[role="dialog"]').contains('button', 'GitHub').should('be.visible')
    cy.get('[role="dialog"]').contains('button', 'HuggingFace').should('be.visible')
    cy.get('[role="dialog"]').contains('button', 'URL').should('be.visible')
  })

  it('switches to GitHub source', () => {
    cy.get('[role="dialog"]').contains('button', 'GitHub').click()
    cy.get('input[placeholder="https://github.com/user/repo"]').should('be.visible')
  })

  it('switches to HuggingFace source', () => {
    cy.get('[role="dialog"]').contains('button', 'HuggingFace').click()
    cy.get('input[placeholder="username/dataset-name"]').should('be.visible')
  })

  it('switches to URL source', () => {
    cy.get('[role="dialog"]').contains('button', 'URL').click()
    cy.get('input[placeholder="https://example.com/data.txt"]').should('be.visible')
  })

  it('shows dataset name input', () => {
    cy.get('input[placeholder="Dataset name (optional)"]').should('be.visible')
  })

  it('has an Import action', () => {
    cy.get('[role="dialog"]').contains('button', /^Import$/).should('be.visible')
  })
})

describe('Dataset cards', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.intercept('GET', `${API}/datasets`, {
      statusCode: 200,
      body: {
        datasets: [
          { id: 'ds1', name: 'shakespeare', source: 'local', size: 1048576, samples: 500, created_at: '2026-08-01T00:00:00Z' },
          { id: 'ds2', name: 'tinyshakespeare', source: 'huggingface', size: 2097152, samples: 1200, created_at: '2026-08-02T00:00:00Z' },
        ],
      },
    }).as('datasetsList')
    cy.intercept('GET', `${API}/datasets/*/versions`, {
      statusCode: 200,
      body: { versions: [], count: 0 },
    }).as('versions')
    cy.visit('/datasets')
  })

  it('shows dataset names', () => {
    cy.contains('shakespeare').should('be.visible')
    cy.contains('tinyshakespeare').should('be.visible')
  })

  it('shows Export buttons on dataset cards', () => {
    cy.get('button[aria-label^="Export"]').should('have.length', 2)
  })

  it('shows Train buttons on dataset cards', () => {
    cy.get('button[aria-label^="Train with"]').should('have.length', 2)
  })
})
