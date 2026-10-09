import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { MetricsCard } from './metrics-card'
import { StatCard } from './stat-card'

describe('MetricsCard', () => {
  it('renders title', () => {
    const html = renderToStaticMarkup(
      <MetricsCard title="Resources">
        <StatCard label="CPU" value="45%" />
      </MetricsCard>
    )
    expect(html).toContain('Resources')
  })

  it('renders children', () => {
    const html = renderToStaticMarkup(
      <MetricsCard title="Section">
        <StatCard label="Memory" value="8 GB" />
      </MetricsCard>
    )
    expect(html).toContain('Memory')
    expect(html).toContain('8 GB')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <MetricsCard title="Test" className="custom-class">
        <StatCard label="A" value="1" />
      </MetricsCard>
    )
    expect(html).toContain('custom-class')
  })

  it('applies titleClassName', () => {
    const html = renderToStaticMarkup(
      <MetricsCard title="Test" titleClassName="text-red-500">
        <StatCard label="A" value="1" />
      </MetricsCard>
    )
    expect(html).toContain('text-red-500')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(
      <MetricsCard title="Test" testId="my-metrics">
        <StatCard label="A" value="1" />
      </MetricsCard>
    )
    expect(html).toContain('data-testid="my-metrics"')
  })

  it('passes columns to KpiGrid', () => {
    const html = renderToStaticMarkup(
      <MetricsCard title="Test" columns={4}>
        <StatCard label="A" value="1" />
        <StatCard label="B" value="2" />
        <StatCard label="C" value="3" />
        <StatCard label="D" value="4" />
      </MetricsCard>
    )
    expect(html).toContain('grid-cols-4')
  })
})
