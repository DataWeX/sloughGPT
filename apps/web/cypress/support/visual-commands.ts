/**
 * Screenshot commands for visual regression testing.
 *
 * Usage:
 *   cy.screenshotPage('home-page')                — full page screenshot
 *   cy.screenshotInteraction('chat', () => {       — screenshot after interaction
 *     cy.get('button').click()
 *   })
 *   cy.screenshotElement('.sidebar', 'sidebar')   — element screenshot
 */

Cypress.Commands.add('screenshotPage', (name: string, options?: { fullPage?: boolean }) => {
  const opts = { fullPage: true, ...options }
  cy.screenshot(`interactions/${name}`, {
    capture: opts.fullPage ? 'fullPage' : 'viewport',
  })
})

Cypress.Commands.add('screenshotInteraction', (name: string, interaction: () => void) => {
  cy.then(() => {
    interaction()
  })
  cy.screenshot(`interactions/${name}`)
})

Cypress.Commands.add('screenshotElement', (selector: string, name: string) => {
  cy.get(selector).screenshot(`interactions/element/${name}`)
})

Cypress.Commands.add('screenshotSequence', (name: string, steps: Array<{ label: string; action: () => void }>) => {
  steps.forEach((step, i) => {
    cy.then(() => {
      step.action()
    })
    cy.screenshot(`interactions/${name}/${i}-${step.label}`)
  })
})
