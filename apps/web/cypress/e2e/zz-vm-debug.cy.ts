describe('home sidebar debug', () => {
  it('dumps sidebar link texts', () => {
    Cypress.on('uncaught:exception', () => false)
    cy.mockAll()
    cy.viewport(1000, 660)
    cy.visit('/')
    cy.wait(1500)
    cy.window().then((w) => {
      const doc = w.document
      const sidebarLinks = Array.from(doc.querySelectorAll('.sl-app-sidebar-desktop a'))
        .map((a) => (a as HTMLAnchorElement).textContent?.trim())
      const startChat = Array.from(doc.querySelectorAll('a'))
        .filter((a) => a.textContent?.includes('Start chatting'))
        .map((a) => {
          const cls = (a as HTMLAnchorElement).className
          const inSidebar = !!a.closest('.sl-app-sidebar-desktop')
          return { inSidebar, cls: String(cls).slice(0, 60), href: (a as HTMLAnchorElement).getAttribute('href') }
        })
      cy.task('writeHydrationDump', JSON.stringify({ sidebarLinks, startChat }))
    })
  })
})
