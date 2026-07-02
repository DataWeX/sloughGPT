import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { CodeSnippet } from './code-snippet'

describe('CodeSnippet', () => {
  it('wraps code in pre with sl-code and scroll by default', () => {
    const html = renderToStaticMarkup(<CodeSnippet>print(1)</CodeSnippet>)
    expect(html).toContain('<pre')
    expect(html).toContain('sl-code')
    expect(html).toContain('overflow-x-auto')
    expect(html).toContain('print(1)')
  })

  it('can disable scroll class', () => {
    const html = renderToStaticMarkup(<CodeSnippet scroll={false}>x</CodeSnippet>)
    expect(html).not.toContain('overflow-x-auto')
  })
})
