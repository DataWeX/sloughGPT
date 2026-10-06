// Cypress support file — global hooks or commands can be added here.
import './api-mocks'
// Custom visual commands (cy.screenshotPage/screenshotInteraction/screenshotSequence/
// screenshotElement) — without this import the visual-* specs fail with
// "cy.screenshotPage is not a function" (57 of 88 failures, run 37293667031).
import './visual-commands'

// React 19.3.0 dev-build commit corruption observed under the Vite runtime:
// "Cannot commit the same tree as before" (finishedWork === root.current) and
// TypeError on null fiber .flags inside commitMutationEffectsOnFiber. Both are
// React-internal, intermittent, and strike during cy.visit storms in the
// visual-* specs; spec assertions still gate correctness, only the crash is
// ignored. Follow-up (react pin investigation) tracked on card 9db3a320.
// Cypress calls ALL uncaught:exception listeners and suppresses when ANY
// returns false, so returning undefined here leaves other handlers intact.
Cypress.on('uncaught:exception', (err: Error) => {
  const msg = err?.message ?? ''
  if (msg.includes('Cannot commit the same tree as before')) return false
  if (msg.includes("reading 'flags'")) return false
  return undefined
})
