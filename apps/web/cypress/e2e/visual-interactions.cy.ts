/**
 * Visual interaction tests — screenshots of basic UI interactions.
 *
 * Run: npx cypress run --spec cypress/e2e/visual-interactions.cy.ts
 * Update baselines: npx cypress run --spec cypress/e2e/visual-interactions.cy.ts --env update=true
 */

describe('Visual interactions — Home page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/')
  })

  it('captures full page', () => {
    cy.screenshotPage('home-full')
  })

  it('captures quick action hover states', () => {
    cy.screenshotSequence('home-quick-actions', [
      { label: 'default', action: () => {} },
      { label: 'hover-chat', action: () => cy.contains('a', 'Start chatting').trigger('mouseover') },
      { label: 'hover-models', action: () => cy.contains('a', 'Personalities').trigger('mouseover') },
      { label: 'hover-training', action: () => cy.contains('a', 'Teach me').trigger('mouseover') },
      { label: 'hover-datasets', action: () => cy.contains('a', 'Datasets').trigger('mouseover') },
    ])
  })

  it('captures sidebar state', () => {
    cy.screenshotPage('home-with-sidebar')
  })
})

describe('Visual interactions — Chat page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/chat')
  })

  it('captures chat empty state', () => {
    cy.screenshotPage('chat-empty')
  })

  it('captures message input focus', () => {
    cy.screenshotInteraction('chat-input-focus', () => {
      cy.get('textarea, input[type="text"]').first().focus()
    })
  })

  it('captures message typed state', () => {
    cy.screenshotInteraction('chat-input-typed', () => {
      cy.get('textarea, input[type="text"]').first().type('Hello, how are you?')
    })
  })
})

describe('Visual interactions — Models page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/models')
  })

  it('captures models grid', () => {
    cy.screenshotPage('models-grid')
  })
})

describe('Visual interactions — Training page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/training')
  })

  it('captures training page', () => {
    cy.screenshotPage('training-page')
  })
})

describe('Visual interactions — Settings page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/settings')
  })

  it('captures settings form', () => {
    cy.screenshotPage('settings-form')
  })
})

describe('Visual interactions — Monitoring page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/monitoring')
  })

  it('captures monitoring dashboard', () => {
    cy.screenshotPage('monitoring-dashboard')
  })
})

describe('Visual interactions — Navigation', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/')
  })

  it('captures full navigation flow', () => {
    cy.screenshotSequence('navigation-flow', [
      { label: '01-home', action: () => {} },
      { label: '02-chat', action: () => cy.visit('/chat') },
      { label: '03-models', action: () => cy.visit('/models') },
      { label: '04-training', action: () => cy.visit('/training') },
      { label: '05-datasets', action: () => cy.visit('/datasets') },
      { label: '06-settings', action: () => cy.visit('/settings') },
      { label: '07-home', action: () => cy.visit('/') },
    ])
  })
})

describe('Visual interactions — Dark mode states', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/')
  })

  it('captures responsive viewport', () => {
    cy.screenshotPage('home-desktop')
    cy.viewport(768, 1024)
    cy.screenshotPage('home-tablet')
    cy.viewport(375, 667)
    cy.screenshotPage('home-mobile')
  })
})
