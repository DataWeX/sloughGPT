import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Chip } from './chip'

describe('Chip', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<Chip>alpha</Chip>)
    expect(html).toContain('alpha')
  })

  it('renders remove control when onRemove is set', () => {
    const html = renderToStaticMarkup(<Chip onRemove={() => {}}>x</Chip>)
    expect(html).toContain('aria-label="Remove"')
  })
})
