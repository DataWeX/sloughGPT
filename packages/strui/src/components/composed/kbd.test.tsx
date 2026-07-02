import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Kbd } from './kbd'

describe('Kbd', () => {
  it('renders keyboard element', () => {
    const html = renderToStaticMarkup(<Kbd>Enter</Kbd>)
    expect(html).toContain('<kbd')
    expect(html).toContain('Enter')
  })
})
