import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { InfoCard } from './info-card'

describe('InfoCard', () => {
  it('renders title', () => {
    const html = renderToStaticMarkup(
      <InfoCard title="GPU Acceleration" />
    )
    expect(html).toContain('GPU Acceleration')
  })

  it('renders description', () => {
    const html = renderToStaticMarkup(
      <InfoCard title="GPU" description="CUDA cores available" />
    )
    expect(html).toContain('CUDA cores available')
  })

  it('renders icon', () => {
    const html = renderToStaticMarkup(
      <InfoCard title="GPU" icon={<span data-testid="icon">GPU</span>} />
    )
    expect(html).toContain('GPU')
  })

  it('renders children', () => {
    const html = renderToStaticMarkup(
      <InfoCard title="GPU">
        <p>Device info here</p>
      </InfoCard>
    )
    expect(html).toContain('Device info here')
  })

  it('applies tone classes', () => {
    const html = renderToStaticMarkup(
      <InfoCard title="GPU" tone="success" icon={<span>X</span>} />
    )
    expect(html).toContain('text-green-500')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(
      <InfoCard title="GPU" testId="gpu-card" />
    )
    expect(html).toContain('data-testid="gpu-card"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <InfoCard title="GPU" className="custom-class" />
    )
    expect(html).toContain('custom-class')
  })
})
