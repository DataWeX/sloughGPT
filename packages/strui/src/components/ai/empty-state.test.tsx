import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { EmptyState } from './empty-state'

describe('EmptyState', () => {
  it('renders title and description', () => {
    const html = renderToStaticMarkup(
      <EmptyState title="Nothing here" description="Try a search" />,
    )
    expect(html).toContain('Nothing here')
    expect(html).toContain('Try a search')
  })

  it('renders children slot', () => {
    const html = renderToStaticMarkup(
      <EmptyState title="T">
        <button type="button">Action</button>
      </EmptyState>,
    )
    expect(html).toContain('Action')
  })
})
