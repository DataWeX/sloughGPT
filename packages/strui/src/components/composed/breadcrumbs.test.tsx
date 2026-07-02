import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Breadcrumbs } from './breadcrumbs'

describe('Breadcrumbs', () => {
  it('renders labeled nav with breadcrumb semantics', () => {
    const html = renderToStaticMarkup(
      <Breadcrumbs items={[{ label: 'Home', href: '/' }, { label: 'Runs' }]} />,
    )
    expect(html).toContain('aria-label="Breadcrumb"')
    expect(html).toContain('Home')
    expect(html).toContain('Runs')
    expect(html).toContain('href="/"')
  })

  it('uses custom separator', () => {
    const html = renderToStaticMarkup(
      <Breadcrumbs separator="›" items={[{ label: 'a' }, { label: 'b' }]} />,
    )
    expect(html).toContain('›')
  })
})
