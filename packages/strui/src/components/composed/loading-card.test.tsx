import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { LoadingCard } from './loading-card'

describe('LoadingCard', () => {
  it('renders without title', () => {
    const html = renderToStaticMarkup(<LoadingCard />)
    expect(html).toContain('animate-pulse')
  })

  it('renders with title', () => {
    const html = renderToStaticMarkup(<LoadingCard title="Health" />)
    expect(html).toContain('Health')
    expect(html).toContain('animate-pulse')
  })

  it('renders specified number of lines', () => {
    const html = renderToStaticMarkup(<LoadingCard lines={4} />)
    // Each line is a Skeleton with h-3 class
    const matches = html.match(/h-3 rounded/g)
    expect(matches?.length).toBeGreaterThanOrEqual(4)
  })

  it('renders height-based skeleton', () => {
    const html = renderToStaticMarkup(<LoadingCard height="h-32" />)
    expect(html).toContain('h-32')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(<LoadingCard testId="loading" />)
    expect(html).toContain('data-testid="loading"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(<LoadingCard className="custom" />)
    expect(html).toContain('custom')
  })
})
