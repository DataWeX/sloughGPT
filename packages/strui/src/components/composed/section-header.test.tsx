import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { SectionHeader } from './section-header'

describe('SectionHeader', () => {
  it('renders title and optional description', () => {
    const html = renderToStaticMarkup(
      <SectionHeader title="Settings" description="Optional copy" />,
    )
    expect(html).toContain('sl-h2')
    expect(html).toContain('Settings')
    expect(html).toContain('Optional copy')
  })

  it('renders action slot', () => {
    const html = renderToStaticMarkup(<SectionHeader title="T" action={<span>Act</span>} />)
    expect(html).toContain('Act')
  })
})
