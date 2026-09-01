/**
 * Visual tests for AppRouteHeader across all pages.
 *
 * Screenshots the header area on every page to catch regressions
 * in typography, spacing, alignment, or design system violations.
 *
 * Run: npx cypress run --spec cypress/e2e/visual-headers.cy.ts
 */

const PAGES = [
  { path: '/', name: 'home' },
  { path: '/chat', name: 'chat' },
  { path: '/models', name: 'models' },
  { path: '/training', name: 'training' },
  { path: '/datasets', name: 'datasets' },
  { path: '/settings', name: 'settings' },
  { path: '/monitoring', name: 'monitoring' },
  { path: '/knowledge', name: 'knowledge' },
  { path: '/memory', name: 'memory' },
  { path: '/shell', name: 'shell' },
  { path: '/vm', name: 'vm' },
  { path: '/companion', name: 'companion' },
  { path: '/agents', name: 'agents' },
  { path: '/collections', name: 'collections' },
  { path: '/souls', name: 'souls' },
  { path: '/tokenizer', name: 'tokenizer' },
  { path: '/token-tree', name: 'token-tree' },
  { path: '/infer', name: 'infer' },
  { path: '/self-train', name: 'self-train' },
  { path: '/lora-eval', name: 'lora-eval' },
  { path: '/meta-weights', name: 'meta-weights' },
  { path: '/rate-limit', name: 'rate-limit' },
  { path: '/benchmark', name: 'benchmark' },
  { path: '/feedback', name: 'feedback' },
  { path: '/images', name: 'images' },
  { path: '/files', name: 'files' },
  { path: '/docstore', name: 'docstore' },
  { path: '/kb', name: 'kb' },
  { path: '/vector', name: 'vector' },
  { path: '/workflow', name: 'workflow' },
  { path: '/world', name: 'world' },
  { path: '/registry', name: 'registry' },
  { path: '/security', name: 'security' },
  { path: '/admin', name: 'admin' },
  { path: '/session', name: 'session' },
  { path: '/evaluate', name: 'evaluate' },
  { path: '/experiments', name: 'experiments' },
  { path: '/export', name: 'export' },
  { path: '/adapters', name: 'adapters' },
  { path: '/learn', name: 'learn' },
  { path: '/voice', name: 'voice' },
  { path: '/multimodal', name: 'multimodal' },
  { path: '/compare', name: 'compare' },
  { path: '/planner', name: 'planner' },
  { path: '/auto-train', name: 'auto-train' },
]

describe('Visual — AppRouteHeader across all pages', () => {
  beforeEach(() => {
    cy.mockAll()
  })

  PAGES.forEach(({ path, name }) => {
    it(`header: ${name} (${path})`, () => {
      cy.visit(path, { failOnStatusCode: false })
      cy.get('body').then(($body) => {
        if ($body.find('[class*="sl-page-header"], [class*="app-route-header"], header, nav').length > 0) {
          cy.get('[class*="sl-page-header"], [class*="app-route-header"], header, nav').first().screenshot(`headers/${name}`)
        } else {
          cy.screenshotPage(`headers/${name}-fullpage`)
        }
      })
    })
  })
})

describe('Visual — Header responsive breakpoints', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/')
  })

  it('header at desktop (1280px)', () => {
    cy.viewport(1280, 800)
    cy.get('[class*="sl-page-header"], [class*="app-route-header"], header, nav').first().screenshot('headers/responsive-desktop')
  })

  it('header at tablet (768px)', () => {
    cy.viewport(768, 1024)
    cy.get('[class*="sl-page-header"], [class*="app-route-header"], header, nav').first().screenshot('headers/responsive-tablet')
  })

  it('header at mobile (375px)', () => {
    cy.viewport(375, 667)
    cy.get('[class*="sl-page-header"], [class*="app-route-header"], header, nav').first().screenshot('headers/responsive-mobile')
  })
})
