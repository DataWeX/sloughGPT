import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { InlineBanner } from './inline-banner'

describe('InlineBanner', () => {
  it('renders status role with title and description', () => {
    const html = renderToStaticMarkup(
      <InlineBanner variant="warning" title="Heads up" description="Details" />,
    )
    expect(html).toContain('role="status"')
    expect(html).toContain('Heads up')
    expect(html).toContain('Details')
    expect(html).toContain('border-warning')
  })

  it('renders action slot', () => {
    const html = renderToStaticMarkup(
      <InlineBanner title="T" action={<button type="button">Retry</button>} />,
    )
    expect(html).toContain('Retry')
  })
})
