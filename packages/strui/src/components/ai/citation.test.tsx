import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Citation } from './citation'

describe('Citation', () => {
  it('renders linked citation when href is set', () => {
    const html = renderToStaticMarkup(
      <Citation index={1} href="https://example.com" title="Doc" />,
    )
    expect(html).toContain('<a ')
    expect(html).toContain('href="https://example.com"')
    expect(html).toContain('title="Doc"')
  })

  it('renders span when no href', () => {
    const html = renderToStaticMarkup(<Citation index={2} />)
    expect(html).toContain('<span')
    expect(html).toContain('title="Source 2"')
  })
})
