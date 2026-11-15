/**
 * Header navigation assertions — the assertion half of the visual specs.
 *
 * visual-headers.cy.ts and visual-interactions.cy.ts CAPTURE headers across
 * pages but never ASSERT anything (screenshots are artifacts; no baseline
 * diffing). This spec proves the interaction journey the header must survive:
 * visit a page → header visible → open nav → click through → header STILL
 * visible on the destination. Screenshots ride along as artifacts only.
 *
 * Raw cy.screenshot() on purpose: cy.screenshotPage/screenshotSequence live
 * in support/visual-commands.ts but are not registered on main (fix
 * 4069194a0), so this spec does not depend on them.
 *
 * Run: npx cypress run --spec cypress/e2e/header-nav.cy.ts
 */

// Screenshot capture races React's dev reconciler the same way
// visual-interactions.cy.ts documents — ignore only those two invariants.
Cypress.on('uncaught:exception', (err) => {
  const msg = typeof err === 'string' ? err : err?.message || String(err)
  return (
    !msg.includes('Cannot commit the same tree as before') &&
    !msg.includes('You cannot render a <Router> inside another <Router>')
  )
})

// Same selector set visual-headers.cy.ts screenshots; AppRouteHeader renders
// a bare <header> (components/AppRouteHeader.tsx), so fall back through the
// app shell's header/nav landmarks.
const HEADER_SELECTOR = '[class*="sl-page-header"], [class*="app-route-header"], header, nav'

const PAGES = [
  { path: '/', name: 'home' },
  { path: '/chat', name: 'chat' },
  { path: '/training', name: 'training' },
  { path: '/datasets', name: 'datasets' },
  { path: '/settings', name: 'settings' },
]

describe('Header — visible and screenshotted across pages', () => {
  beforeEach(() => {
    cy.viewport(1280, 800)
    cy.mockAll()
  })

  PAGES.forEach(({ path, name }) => {
    it(`header visible on ${name} (${path})`, () => {
      cy.visit(path, { failOnStatusCode: false })
      // First visible header landmark — a real assertion, not a capture.
      cy.get(HEADER_SELECTOR).filter(':visible').first().should('exist')
      cy.get(HEADER_SELECTOR).filter(':visible').first().screenshot(`header-nav/${name}-header`)
    })
  })
})

describe('Header — survives desktop nav click-through', () => {
  beforeEach(() => {
    // .sl-app-sidebar-desktop only renders >=1024px (AppLayout.tsx).
    cy.viewport(1280, 800)
    cy.mockAll()
    cy.visit('/')
  })

  it('sidebar visible, click-through to /chat keeps header visible', () => {
    cy.get('.sl-app-sidebar-desktop').should('be.visible')
    cy.get(HEADER_SELECTOR).filter(':visible').first().screenshot('header-nav/before-click')
    // Sidebar labels are i18n-keyed (lib/navigation.ts); click by href, not label.
    cy.get('.sl-app-sidebar-desktop a[href="/chat"]').first().click()
    cy.location('pathname').should('eq', '/chat')
    // The header must still be there after the route change.
    cy.get(HEADER_SELECTOR).filter(':visible').first().should('exist')
    cy.get('.sl-app-content, main').filter(':visible').first().should('exist')
    cy.get(HEADER_SELECTOR).filter(':visible').first().screenshot('header-nav/after-click-chat')
  })
})

describe('Header — mobile menu opens and navigates', () => {
  beforeEach(() => {
    cy.viewport(375, 667)
    cy.mockAll()
    cy.visit('/')
  })

  it('mobile header menu button opens drawer; header stays visible', () => {
    // AppLayout.tsx: <header class="sl-app-mobile-header"> carries the
    // aria-expanded menu button whose sr-only label is "Open menu".
    cy.get('header.sl-app-mobile-header').should('be.visible')
    cy.get('header.sl-app-mobile-header').screenshot('header-nav/mobile-header')
    cy.contains('button', 'Open menu').click()
    cy.get('#mobile-navigation-drawer').should('be.visible')
    cy.get('#mobile-navigation-drawer').screenshot('header-nav/mobile-drawer-open')
    cy.get('header.sl-app-mobile-header').should('be.visible')
  })
})
