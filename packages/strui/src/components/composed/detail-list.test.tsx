import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { DetailList } from './detail-list'

describe('DetailList', () => {
  const items = [
    { label: 'Model', value: 'GPT-2' },
    { label: 'Path', value: '/models/gpt2', mono: true },
    { label: 'Size', value: '500 MB' },
  ]

  it('renders items in row layout', () => {
    const html = renderToStaticMarkup(<DetailList items={items} />)
    expect(html).toContain('Model')
    expect(html).toContain('GPT-2')
    expect(html).toContain('Path')
  })

  it('renders items in grid layout', () => {
    const html = renderToStaticMarkup(<DetailList items={items} layout="grid" />)
    expect(html).toContain('grid-cols-2')
    expect(html).toContain('Model')
  })

  it('applies mono class', () => {
    const html = renderToStaticMarkup(<DetailList items={items} />)
    expect(html).toContain('font-mono')
  })

  it('renders as link when href provided', () => {
    const itemsWithLink = [{ label: 'Docs', value: 'View', href: 'https://example.com' }]
    const html = renderToStaticMarkup(<DetailList items={itemsWithLink} />)
    expect(html).toContain('href="https://example.com"')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(<DetailList items={items} testId="details" />)
    expect(html).toContain('data-testid="details"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(<DetailList items={items} className="custom" />)
    expect(html).toContain('custom')
  })

  it('renders grid with different column counts', () => {
    const html3 = renderToStaticMarkup(<DetailList items={items} layout="grid" columns={3} />)
    expect(html3).toContain('grid-cols-3')
    const html4 = renderToStaticMarkup(<DetailList items={items} layout="grid" columns={4} />)
    expect(html4).toContain('grid-cols-4')
  })
})
