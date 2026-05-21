import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { PageHeader } from './page-header'

describe('PageHeader', () => {
  it('renders header with sl-h1 title', () => {
    const html = renderToStaticMarkup(<PageHeader title="Models" />)
    expect(html).toContain('<header')
    expect(html).toContain('sl-h1')
    expect(html).toContain('Models')
  })

  it('renders description and actions', () => {
    const html = renderToStaticMarkup(
      <PageHeader title="T" description="Sub" actions={<span>Go</span>} />,
    )
    expect(html).toContain('Sub')
    expect(html).toContain('Go')
  })
})
