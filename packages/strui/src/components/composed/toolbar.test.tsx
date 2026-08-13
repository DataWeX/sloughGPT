import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Toolbar } from './toolbar'

describe('Toolbar', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<Toolbar><button>Filter</button></Toolbar>)
    expect(html).toContain('<button')
    expect(html).toContain('Filter')
  })

  it('renders a toolbar role', () => {
    const html = renderToStaticMarkup(<Toolbar>x</Toolbar>)
    expect(html).toContain('role="toolbar"')
  })

  it('passes className to the wrapper', () => {
    const html = renderToStaticMarkup(<Toolbar className="my-toolbar">x</Toolbar>)
    expect(html).toContain('my-toolbar')
  })

  it('passes through extra props', () => {
    const html = renderToStaticMarkup(<Toolbar aria-label="Actions">x</Toolbar>)
    expect(html).toContain('aria-label="Actions"')
  })
})
