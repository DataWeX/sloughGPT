describe('Export page', () => {
  beforeEach(() => {
    cy.mockExport()
    cy.visit('/export')
  })

  it('renders the page', () => {
    cy.contains('Export').should('exist')
  })

  it('shows loaded model', () => {
    cy.contains('gpt2').should('exist')
  })

  it('shows format buttons', () => {
    cy.contains('SOU').should('exist')
    cy.contains('ONNX').should('exist')
    cy.contains('GGUF').should('exist')
  })

  it('can select a different format', () => {
    cy.contains('ONNX').click()
    cy.contains('ONNX').should('have.class', 'bg-primary')
  })

  it('exports the model', () => {
    cy.contains('Export as').click()
    cy.wait('@modelExport')
    cy.contains('Exported Files').should('exist')
  })
})
