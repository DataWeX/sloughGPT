import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { InsightsCard } from './insights-card'

describe('InsightsCard', () => {
  it('renders title', () => {
    const html = renderToStaticMarkup(<InsightsCard title="Feedback Insights" />)
    expect(html).toContain('Feedback Insights')
  })

  it('renders KPIs', () => {
    const html = renderToStaticMarkup(
      <InsightsCard
        title="Insights"
        kpis={[
          { label: 'Sentiment', value: '85%' },
          { label: 'Quality', value: 'Excellent' },
        ]}
      />
    )
    expect(html).toContain('Sentiment')
    expect(html).toContain('85%')
    expect(html).toContain('Quality')
  })

  it('renders details', () => {
    const html = renderToStaticMarkup(
      <InsightsCard
        title="Details"
        details={[
          { label: 'Total conversations', value: 42 },
          { label: 'Total messages', value: 128 },
        ]}
      />
    )
    expect(html).toContain('Total conversations')
    expect(html).toContain('42')
    expect(html).toContain('Total messages')
  })

  it('renders children', () => {
    const html = renderToStaticMarkup(
      <InsightsCard title="Test">
        <p>Extra content</p>
      </InsightsCard>
    )
    expect(html).toContain('Extra content')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(
      <InsightsCard title="Test" testId="my-insights" />
    )
    expect(html).toContain('data-testid="my-insights"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <InsightsCard title="Test" className="custom-card" />
    )
    expect(html).toContain('custom-card')
  })

  it('renders without kpis or details', () => {
    const html = renderToStaticMarkup(<InsightsCard title="Empty" />)
    expect(html).toContain('Empty')
  })
})
