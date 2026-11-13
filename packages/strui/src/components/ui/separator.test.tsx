import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Separator } from './separator'

describe('Separator', () => {
  it('defaults to decorative (role none)', () => {
    const html = renderToStaticMarkup(<Separator />)
    expect(html).toContain('role="none"')
    expect(html).not.toContain('role="separator"')
  })

  it('renders separator role when not decorative', () => {
    const html = renderToStaticMarkup(<Separator decorative={false} />)
    expect(html).toContain('role="separator"')
  })

  it('defaults to horizontal orientation', () => {
    const html = renderToStaticMarkup(<Separator decorative={false} />)
    expect(html).toContain('aria-orientation="horizontal"')
  })

  it('renders vertical orientation', () => {
    const html = renderToStaticMarkup(<Separator orientation="vertical" decorative={false} />)
    expect(html).toContain('aria-orientation="vertical"')
  })

  it('applies horizontal size classes', () => {
    const html = renderToStaticMarkup(<Separator />)
    expect(html).toContain('h-px')
    expect(html).toContain('w-full')
  })

  it('applies vertical size classes', () => {
    const html = renderToStaticMarkup(<Separator orientation="vertical" />)
    expect(html).toContain('h-full')
    expect(html).toContain('w-px')
  })

  it('merges custom className', () => {
    const html = renderToStaticMarkup(<Separator className="my-sep" />)
    expect(html).toContain('my-sep')
  })

  it('passes extra props to the element', () => {
    const html = renderToStaticMarkup(<Separator data-testid="sep" />)
    expect(html).toContain('data-testid="sep"')
  })
})
