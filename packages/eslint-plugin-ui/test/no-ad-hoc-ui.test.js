/**
 * RuleTester suite for no-ad-hoc-ui — runs standalone, no test framework:
 *     node packages/eslint-plugin-ui/test/no-ad-hoc-ui.test.js
 * eslint's RuleTester throws on any mismatch; a clean run prints and exits 0.
 */
'use strict'

const { RuleTester } = require('eslint')
const rule = require('../rules/no-ad-hoc-ui')

const ruleTester = new RuleTester({
  parser: require.resolve('@typescript-eslint/parser'),
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
  },
})

const APP_FILE = '/repo/apps/web/components/Foo.tsx'
const TEST_FILE = '/repo/apps/web/components/Foo.test.tsx'

ruleTester.run('no-ad-hoc-ui', rule, {
  valid: [
    // strui element — the design-system path
    { code: '<Button onClick={save}>Resend</Button>', filename: APP_FILE },
    // modal backdrop: click-outside-to-close, presentational
    { code: '<div aria-hidden="true" onClick={onClose} />', filename: APP_FILE },
    // row content wrapper: bare stopPropagation, not an interactive element
    {
      code: '<div onClick={(e) => e.stopPropagation()}><span>x</span></div>',
      filename: APP_FILE,
    },
    // tokens, not literals
    {
      code: '<div style={{ color: "var(--color-primary)" }}>x</div>',
      filename: APP_FILE,
    },
    { code: '<span className="bg-primary text-on-primary">x</span>', filename: APP_FILE },
    // non-JSX string that only looks hex-adjacent
    { code: 'const label = "#face of the moon"', filename: APP_FILE },
    // test files are the mock layer — skipped by design
    { code: '<button onClick={x}>mock</button>', filename: TEST_FILE },
    // no JSX at all
    { code: 'export const n = 1', filename: APP_FILE },
    // value-less aria-hidden is presence-true — still a backdrop
    { code: '<div aria-hidden onClick={onClose} />', filename: APP_FILE },
  ],

  invalid: [
    // 1. raw button markup
    {
      code: '<button type="button" onClick={x}>Save</button>',
      filename: APP_FILE,
      errors: [{ messageId: 'rawButton' }],
    },
    // 2. fake buttons (backdrop and stopPropagation cases do NOT rescue these)
    {
      code: '<div className="row" onClick={() => toggle(i)} />',
      filename: APP_FILE,
      errors: [{ messageId: 'fakeButton' }],
    },
    {
      code: '<span onClick={onClose}>x</span>',
      filename: APP_FILE,
      errors: [{ messageId: 'fakeButton' }],
    },
    // 3. hardcoded colors — attr form, style-object form, plain literal form
    {
      code: '<svg><path fill="#ff0000" /></svg>',
      filename: APP_FILE,
      errors: [{ messageId: 'hardColor' }],
    },
    {
      code: '<div style={{ backgroundColor: "#1a2b3c" }}>x</div>',
      filename: APP_FILE,
      errors: [{ messageId: 'hardColor' }],
    },
    {
      code: 'const CANVAS_BG = "#abc"; draw(CANVAS_BG)',
      filename: APP_FILE,
      errors: [{ messageId: 'hardColor' }],
    },
    {
      code: '<div title="#ffffff" onClick={fn} />',
      filename: APP_FILE,
      errors: [{ messageId: 'fakeButton' }, { messageId: 'hardColor' }],
    },
  ],
})

console.log('no-ad-hoc-ui RuleTester: all cases pass')
