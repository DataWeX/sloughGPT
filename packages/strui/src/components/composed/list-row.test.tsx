import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ListRow } from './list-row'

describe('ListRow', () => {
  it('renders with label', () => {
    const html = renderToStaticMarkup(<ListRow label="Conversation A" />)
    expect(html).toContain('Conversation A')
  })

  it('renders value and action', () => {
    const html = renderToStaticMarkup(
      <ListRow label="T" value="Updated" action="›" />,
    )
    expect(html).toContain('Updated')
    expect(html).toContain('›')
  })
})
