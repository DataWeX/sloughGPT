describe('hydration diagnostic', () => {
  it('captures post-hydration head and body', () => {
    Cypress.on('uncaught:exception', () => false)
    cy.visit('/')
    cy.wait(2500)
    cy.document().then((doc) => {
      const head = doc.head ? doc.head.innerHTML : 'NO HEAD'
      const body = doc.body ? doc.body.innerHTML.slice(0, 2000) : 'NO BODY'
      cy.task('writeHydrationDump', JSON.stringify({ head, body }, null, 2))
    })
  })
})
