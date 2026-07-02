import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { FoldSection } from './fold-section'

describe('FoldSection', () => {
  it('renders details with summary heading and body', () => {
    const html = renderToStaticMarkup(
      <FoldSection heading="More options">
        <p>Hidden until expanded</p>
      </FoldSection>,
    )
    expect(html).toContain('<details')
    expect(html).toContain('More options')
    expect(html).toContain('Hidden until expanded')
  })
})
