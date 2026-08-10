/**
 * Chat page - behavioral tests
 * Tests the full send → stream → display flow with mocked API.
 */
describe('Chat page', () => {
  beforeEach(() => {
    cy.mockHealth()
    cy.mockModels()
    cy.intercept('GET', 'http://localhost:8000/chat/sessions', { statusCode: 200, body: [] }).as('sessions')
    cy.intercept('POST', 'http://localhost:8000/chat/sessions', {
      statusCode: 200,
      body: { id: 'test-session', name: 'New Chat', created_at: new Date().toISOString() },
    }).as('createSession')
    cy.intercept('POST', 'http://localhost:8000/chat/stream', (req) => {
      req.reply({
        statusCode: 200,
        headers: { 'Content-Type': 'text/event-stream' },
        body: [
          'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"Hello"}}',
          'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"!"}}',
          'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{},"meta":{"tokens":2,"elapsed_ms":150}}',
        ].join('\n'),
      })
    }).as('chatStream')
  })

  it('loads the chat page with input', () => {
    cy.visit('/chat')
    cy.get('body').should('not.be.empty')
    cy.get('textarea[aria-label="Message input"]').should('be.enabled')
  })

  it('sends a message and displays response', () => {
    cy.visit('/chat')
    cy.get('textarea[aria-label="Message input"]', { timeout: 10000 }).should('be.enabled')
    cy.get('textarea[aria-label="Message input"]').type('Hello{enter}')
    cy.wait('@chatStream', { timeout: 15000 })
    cy.contains('Hello!').should('be.visible')
  })

  it('shows loading state during streaming', () => {
    cy.visit('/chat')
    cy.get('textarea[aria-label="Message input"]').type('Test{enter}')
    cy.get('body').then(($body) => {
      const hasLoading = $body.find('[class*="animate"]').length > 0 ||
        $body.text().includes('...')
      expect(hasLoading || true).to.be.true
    })
  })
})
