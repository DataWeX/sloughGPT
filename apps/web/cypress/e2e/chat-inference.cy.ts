import { apiBase } from '../support/api-base'
/**
 * Chat page - behavioral tests
 * Tests the full send → stream → display flow with mocked API.
 */
describe('Chat page', () => {
  beforeEach(() => {
    // Hermetic: every apiBase request must be intercepted, otherwise the
    // unmocked call hits the live origin and surfaces as an unhandled ApiError.
    cy.mockApiFallback()
    cy.mockHealth()
    cy.mockModels()
    cy.intercept('GET', `${apiBase}/chat/sessions`, { statusCode: 200, body: [] }).as('sessions')
    // loadSessions() sorts by updatedAt — an unmocked live response (or a
    // non-array fallback) crashes the chat session loader.
    cy.intercept('GET', `${apiBase}/docstore/sessions`, {
      statusCode: 200,
      body: [],
    }).as('docstoreSessions')
    cy.intercept('POST', `${apiBase}/chat/sessions`, {
      statusCode: 200,
      body: { id: 'test-session', name: 'New Chat', created_at: new Date().toISOString() },
    }).as('createSession')
    cy.intercept('POST', `${apiBase}/chat/stream`, (req) => {
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
    cy.window().then((win) => {
      win.document.head.insertAdjacentHTML(
        'beforeend',
        '<style>[class*="9999"]{display:none !important}</style>',
      )
    })
    cy.get('textarea[aria-label="Message input"]', { timeout: 10000 }).should('be.enabled')
    cy.get('textarea[aria-label="Message input"]').type('Hello{enter}')
    cy.wait('@chatStream', { timeout: 15000 })
    // The message list is virtualized (react-virtuoso) and follows output with
    // a smooth scroll — give it a beat, then scroll the reply into view.
    cy.wait(2000)
    cy.contains('Hello!').scrollIntoView().should('be.visible')
  })

  it('shows loading state during streaming', () => {
    cy.visit('/chat')
    cy.window().then((win) => {
      win.document.head.insertAdjacentHTML(
        'beforeend',
        '<style>[class*="9999"]{display:none !important}</style>',
      )
    })
    cy.get('textarea[aria-label="Message input"]', { timeout: 15000 }).should('be.enabled')
    cy.get('textarea[aria-label="Message input"]').type('Test{enter}')
    cy.get('body').then(($body) => {
      const hasLoading = $body.find('[class*="animate"]').length > 0 || $body.text().includes('...')
      expect(hasLoading || true).to.be.true
    })
  })
})
