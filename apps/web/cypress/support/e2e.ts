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
//
// ONLY the uncaught:exception paths may be suppressed — they are safe by
// construction. Cypress 14.5.4 dispatches them as
//     if (lodash.some(results, returnedFalse)) { ... }
// i.e. suppression requires a listener to actually RETURN false; a listener
// returning undefined changes nothing for the other handlers.
//
// *** DO NOT register a 'fail' listener here. ***
// The cy:fail path is structurally different (cypress_runner.js):
//     rets = this.Cypress.action('cy:fail', err, runnable)
//     if (rets && rets.length) { return }   // "bail if we had callbacks attached"
// emitMap returns ONE entry PER LISTENER, including listeners that returned
// undefined — so merely REGISTERING 'fail' makes rets non-empty and swallows
// EVERY test failure in the suite, regardless of what the handler returns.
// That ternary below never mattered: undefined still lands in rets. Proven
// with three deliberate controls (plain throw / chai assertion / cy.get
// timeout) all reporting green — card 20261125_130530. There is no way to
// register a selective fail listener, because the bail is on LENGTH, not on
// the returned values.
//
// Cost of removing it: the React commit-corruption crash is no longer
// suppressed when it surfaces as a converted test failure (it still is, on
// both uncaught:exception paths above). That trade is deliberate — an
// invisible test suite is strictly worse than an intermittent crash — and
// the durable fix is the React pin on card 9db3a320.
const isReactCommitCorruption = (err: unknown): boolean => {
  const msg = typeof err === 'string' ? err : ((err as Error | undefined)?.message ?? '')
  return msg.includes('Cannot commit the same tree as before') || msg.includes("reading 'flags'")
}

Cypress.on('uncaught:exception', (err: Error) => (isReactCommitCorruption(err) ? false : undefined))

beforeEach(() => {
  cy.on('uncaught:exception', (err: Error) => (isReactCommitCorruption(err) ? false : undefined))
})
