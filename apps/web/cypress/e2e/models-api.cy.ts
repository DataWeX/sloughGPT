/**
 * Models page - behavioral tests
 * Tests model listing, loading, and health display.
 */
describe('Models page', () => {
  beforeEach(() => {
    cy.mockHealth({ model_loaded: true, model_type: 'gpt2' })
    cy.mockSystem()
    cy.intercept('GET', 'http://localhost:8000/models/hf', {
      statusCode: 200,
      body: [
        { id: 'gpt2', name: 'GPT-2', source: 'huggingface', loaded: true, size_gb: 0.5, parameters: '124M' },
        { id: 'gpt2-medium', name: 'GPT-2 Medium', source: 'huggingface', loaded: false, size_gb: 1.5, parameters: '355M' },
      ],
    }).as('modelsHf')
    cy.intercept('GET', 'http://localhost:8000/souls', {
      statusCode: 200,
      body: [],
    }).as('souls')
    cy.intercept('GET', 'http://localhost:8000/souls/current', {
      statusCode: 200,
      body: { name: 'default', description: 'Default personality' },
    }).as('currentSoul')
  })

  it('displays model list from API', () => {
    cy.visit('/models')
    cy.wait('@modelsHf')
    cy.contains('GPT-2').should('be.visible')
    cy.contains('GPT-2 Medium').should('be.visible')
  })

  it('shows loaded model indicator', () => {
    cy.visit('/models')
    cy.wait('@modelsHf')
    cy.contains('GPT-2').should('be.visible')
    cy.get('body').then(($body) => {
      const text = $body.text()
      expect(text).to.satisfy((t: string) => t.includes('Loaded') || t.includes('loaded') || t.includes('Active'))
    })
  })

  it('shows model sizes', () => {
    cy.visit('/models')
    cy.wait('@modelsHf')
    cy.get('body').then(($body) => {
      const text = $body.text()
      expect(text).to.satisfy((t: string) => t.includes('GB') || t.includes('0.5') || t.includes('124M'))
    })
  })
})
