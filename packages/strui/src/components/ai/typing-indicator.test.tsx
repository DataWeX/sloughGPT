import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { TypingIndicator } from './typing-indicator'

describe('TypingIndicator', () => {
  it('exposes polite live region and default label', () => {
    const html = renderToStaticMarkup(<TypingIndicator />)
    expect(html).toContain('role="status"')
    expect(html).toContain('aria-live="polite"')
    expect(html).toContain('Assistant is responding')
    expect(html).toContain('str-dot')
  })

  it('accepts custom label', () => {
    const html = renderToStaticMarkup(<TypingIndicator label="Model thinking" />)
    expect(html).toContain('aria-label="Model thinking"')
  })
})
