/**
 * @sloughgpt/eslint-plugin-ui — design-system conformance rules.
 *
 * Plugin key '@sloughgpt/ui' (resolves as @sloughgpt/eslint-plugin-ui).
 * Projector, not authority: the enforcement contract is
 * tests/contract/web_design_system_baseline.json, checked in CI by the
 * Python floor (tests/contract/test_web_design_system.py). These rules
 * surface the same violation SITES at editor time.
 */
'use strict'

module.exports = {
  rules: {
    'no-ad-hoc-ui': require('./rules/no-ad-hoc-ui'),
  },
}
