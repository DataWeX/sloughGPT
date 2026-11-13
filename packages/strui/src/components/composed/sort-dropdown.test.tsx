import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { SortDropdown } from './sort-dropdown'

describe('SortDropdown', () => {
  const options = [
    { value: 'newest', label: 'Newest' },
    { value: 'oldest', label: 'Oldest' },
    { value: 'importance', label: 'Importance' },
  ]

  it('renders with current sort label', () => {
    const html = renderToStaticMarkup(
      <SortDropdown value="newest" options={options} onChange={() => {}} />
    )
    expect(html).toContain('Newest')
    expect(html).toContain('Sort by')
  })

  it('renders custom label', () => {
    const html = renderToStaticMarkup(
      <SortDropdown value="oldest" options={options} onChange={() => {}} label="Order" />
    )
    expect(html).toContain('Order')
    expect(html).toContain('Oldest')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(
      <SortDropdown value="newest" options={options} onChange={() => {}} testId="sort-menu" />
    )
    expect(html).toContain('data-testid="sort-menu"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <SortDropdown value="newest" options={options} onChange={() => {}} className="custom-class" />
    )
    expect(html).toContain('custom-class')
  })
})
