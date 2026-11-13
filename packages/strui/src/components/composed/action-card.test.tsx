import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ActionCard } from './action-card'

describe('ActionCard', () => {
  it('renders title', () => {
    const html = renderToStaticMarkup(
      <ActionCard title="System Health">
        <p>Content</p>
      </ActionCard>
    )
    expect(html).toContain('System Health')
  })

  it('renders subtitle', () => {
    const html = renderToStaticMarkup(
      <ActionCard title="Metrics" subtitle="Last updated 2m ago">
        <p>Content</p>
      </ActionCard>
    )
    expect(html).toContain('Last updated 2m ago')
  })

  it('renders actions', () => {
    const html = renderToStaticMarkup(
      <ActionCard
        title="Card"
        actions={<button>Refresh</button>}
      >
        <p>Content</p>
      </ActionCard>
    )
    expect(html).toContain('Refresh')
  })

  it('renders children', () => {
    const html = renderToStaticMarkup(
      <ActionCard title="Card">
        <p>Body content here</p>
      </ActionCard>
    )
    expect(html).toContain('Body content here')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(
      <ActionCard title="Card" testId="my-card">
        <p>Content</p>
      </ActionCard>
    )
    expect(html).toContain('data-testid="my-card"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <ActionCard title="Card" className="custom-class">
        <p>Content</p>
      </ActionCard>
    )
    expect(html).toContain('custom-class')
  })
})
