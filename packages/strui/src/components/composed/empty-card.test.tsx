import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { EmptyCard } from './empty-card'

describe('EmptyCard', () => {
  it('renders dashed card with default message', () => {
    const html = renderToStaticMarkup(<EmptyCard />)
    expect(html).toContain('border-dashed')
    expect(html).toContain('No items')
  })

  it('renders custom message and action', () => {
    const html = renderToStaticMarkup(
      <EmptyCard message="Custom message" action={<button>Click</button>} />,
    )
    expect(html).toContain('Custom message')
    expect(html).toContain('Click')
  })
})
