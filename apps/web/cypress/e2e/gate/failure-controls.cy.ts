/**
 * EXPECTED-FAILURE GATE (card 20261125_130530) — these tests are DESIGNED to
 * fail, so CI asserts exit != 0 on this directory rather than exit 0.
 *
 * Lives under cypress/e2e/gate/ and is excluded from the normal `cypress run`
 * by excludeSpecPattern (cypress.config.ts). The dedicated CI step
 * "Fail-suppression gate (must FAIL)" runs it through cypress.gate.config.ts
 * (a separate config because excludeSpecPattern filters an explicit `--spec`
 * too, which would make the gate pass by finding zero specs) and turns the
 * job red if it ever reports green — or if any control fails to run.
 *
 * Why the inversion exists: Cypress 14.5.4's cy:fail dispatch is
 *     rets = this.Cypress.action('cy:fail', err, runnable)
 *     if (rets && rets.length) { return }   // "bail if we had callbacks attached"
 * and emitMap returns ONE entry PER LISTENER including handlers that returned
 * undefined — so merely REGISTERING Cypress.on('fail') swallows EVERY failure.
 * (uncaught:exception is different: lodash.some(results, returnedFalse) needs a
 * real `false`, which is why the two uncaught handlers in support/e2e.ts are
 * safe and were kept.)
 *
 * This spec has NO record-only listeners of its own — only the global handlers
 * in support/e2e.ts — and drives the three distinct Cypress failure paths:
 *   1. plain throw inside cy.then
 *   2. chai assertion failure
 *   3. cy.get command timeout
 * Expected: all three FAIL (exit 1). A green one means suppression returned.
 */
describe('Failure path controls', () => {
  it('control-1: plain throw must fail', () => {
    cy.then(() => {
      throw new Error('PLAIN_FAILURE_SENTINEL')
    })
  })

  it('control-2: assertion failure must fail', () => {
    cy.then(() => {
      expect(false, 'ASSERTION_FAILURE_SENTINEL').to.equal(true)
    })
  })

  it('control-3: cy.get timeout must fail', () => {
    cy.get('.definitely-not-here-xyz', { timeout: 3000 })
  })
})
