import { renderToStaticMarkup } from 'react-dom/server'
import type { ComponentType } from 'react'
import { describe, expect, it } from 'vitest'

import * as icons from './icons'

const iconEntries = Object.entries(icons).filter(([name]) => name.startsWith('Icon')) as [
  string,
  ComponentType<{ className?: string }>,
][]

describe('icons module', () => {
  it('exports a non-empty set of Icon* components', () => {
    expect(iconEntries.length).toBeGreaterThan(0)
  })

  it('includes the common icon set', () => {
    for (const name of ['IconSearch', 'IconStar', 'IconPin', 'IconMore', 'IconCog', 'IconModels']) {
      expect(iconEntries.some(([n]) => n === name)).toBe(true)
    }
  })
})

for (const [name, Component] of iconEntries) {
  describe(name, () => {
    it('renders an svg with a 24x24 viewBox and currentColor stroke or fill', () => {
      const html = renderToStaticMarkup(<Component />)
      expect(html).toContain('<svg')
      expect(html).toContain('viewBox="0 0 24 24"')
      const hasStroke = html.includes('stroke="currentColor"')
      const hasFill = html.includes('fill="currentColor"')
      expect(hasStroke || hasFill).toBe(true)
    })

    it('passes className through to the svg', () => {
      const html = renderToStaticMarkup(<Component className="my-icon" />)
      expect(html).toContain('class="my-icon"')
    })
  })
}

describe('filled variants', () => {
  it('IconStar renders a filled star when filled is true', () => {
    const html = renderToStaticMarkup(<icons.IconStar filled />)
    expect(html).toContain('fill="currentColor"')
    expect(html).not.toContain('stroke=')
  })

  it('IconStar renders an outline star by default', () => {
    const html = renderToStaticMarkup(<icons.IconStar />)
    expect(html).toContain('stroke="currentColor"')
  })

  it('IconHeart renders a filled heart when filled is true', () => {
    const html = renderToStaticMarkup(<icons.IconHeart filled />)
    expect(html).toContain('fill="currentColor"')
    expect(html).not.toContain('stroke=')
  })

  it('IconHeart renders an outline heart by default', () => {
    const html = renderToStaticMarkup(<icons.IconHeart />)
    expect(html).toContain('stroke="currentColor"')
  })
})

describe('fill-based icons', () => {
  it('IconPin is fill-based', () => {
    const html = renderToStaticMarkup(<icons.IconPin />)
    expect(html).toContain('fill="currentColor"')
    expect(html).not.toContain('stroke=')
  })

  it('IconMore is fill-based', () => {
    const html = renderToStaticMarkup(<icons.IconMore />)
    expect(html).toContain('fill="currentColor"')
    expect(html).not.toContain('stroke=')
  })
})

describe('alias icons', () => {
  it('IconCog renders identically to IconSettings', () => {
    expect(renderToStaticMarkup(<icons.IconCog className="x" />)).toBe(
      renderToStaticMarkup(<icons.IconSettings className="x" />),
    )
  })

  it('IconModels renders identically to IconModel', () => {
    expect(renderToStaticMarkup(<icons.IconModels className="x" />)).toBe(
      renderToStaticMarkup(<icons.IconModel className="x" />),
    )
  })
})
