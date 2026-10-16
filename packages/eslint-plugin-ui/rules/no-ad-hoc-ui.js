/**
 * no-ad-hoc-ui — report design-system bypasses in web UI source.
 *
 * The enforcement contract is tests/contract/web_design_system_baseline.json
 * (per-file counts, frozen-shrink-only), checked in CI by the Python floor
 * tests/contract/test_web_design_system.py. This rule is the editor-time
 * PROJECTOR of the same shape rules; it reports offending sites at 'warn'
 * severity and deliberately does NOT compare counts, so there is exactly one
 * authority for "how much is too much".
 *
 * Shape rules (kept in lockstep with the Python floor):
 *   1. raw <button> markup            -> use @sloughgpt/strui Button
 *   2. div/span with onClick          -> a fake button, unless the tag is a
 *      modal backdrop (aria-hidden) or a row content wrapper (bare
 *      stopPropagation handler)
 *   3. hex color string literals      -> Noir Violet tokens; canvas/chart
 *      renderers are frozen-baseline in the floor, still surfaced here
 *
 * Known v1 gaps (documented, deliberate): hex inside template literals is
 * not visited; test files are skipped by filename, not by override config.
 */
'use strict'

// Longest-first alternation with lookahead so '#ffffff' is one hit and never
// a partial '#fff' hit — byte-for-byte the floor's COLOR_RE.
const HEX_RE = /#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])/

module.exports = {
  meta: {
    type: 'suggestion',
    docs: {
      description:
        'forbid ad-hoc UI markup (raw <button>, div/span onClick, hardcoded hex colors) in favor of @sloughgpt/strui design-system elements',
    },
    schema: [],
    messages: {
      rawButton:
        'raw <button> markup — use the @sloughgpt/strui Button so the design-system chrome applies',
      fakeButton:
        '<{{tag}} onClick> is a fake button — no semantics, no keyboard; use strui Button (backdrops: add aria-hidden)',
      hardColor:
        'hardcoded color {{color}} — use a Noir Violet token (canvas renderers are baseline-frozen, not exempt)',
    },
  },

  create(context) {
    const filename = context.getFilename()
    if (filename.includes('.test.')) return {}

    const sourceCode = context.getSourceCode()

    function onClickAttr(node) {
      return node.attributes.find((a) => a.type === 'JSXAttribute' && a.name.name === 'onClick')
    }

    function hasAriaHidden(node) {
      return node.attributes.some((a) => a.type === 'JSXAttribute' && a.name.name === 'aria-hidden')
    }

    function onClickIsStopPropagation(attr) {
      if (!attr.value) return false
      return sourceCode.getText(attr.value).includes('stopPropagation')
    }

    return {
      JSXOpeningElement(node) {
        const tag = node.name.name
        if (typeof tag !== 'string') return // member expression (<Foo.Bar>)

        if (tag === 'button') {
          context.report({ node, messageId: 'rawButton' })
          return
        }

        if ((tag === 'div' || tag === 'span') && !hasAriaHidden(node)) {
          const attr = onClickAttr(node)
          if (attr && !onClickIsStopPropagation(attr)) {
            context.report({ node, messageId: 'fakeButton', data: { tag } })
          }
        }
      },

      Literal(node) {
        if (typeof node.value !== 'string') return
        const match = node.value.match(HEX_RE)
        if (match) {
          context.report({
            node,
            messageId: 'hardColor',
            data: { color: match[0] },
          })
        }
      },
    }
  },
}
