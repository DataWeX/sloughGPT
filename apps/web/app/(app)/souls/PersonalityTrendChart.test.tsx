import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PersonalityTrendChart } from './PersonalityTrendChart'

/**
 * Design-contract test for the Personality Evolution chart (card 180aae13).
 * Renders with REAL recharts (not mocked) so prop/type regressions surface,
 * and pins the Behance-informed presentation rules:
 *  - Label-role eyebrow title (uppercase, tracked)
 *  - inline legend with current-value readout
 *  - interaction-count caption
 *  - chart-token colors only (no hex anywhere)
 */

const LABELS = { warmth: 'Warmth', confidence: 'Confidence' }

const HISTORY = [
  { timestamp: 1759650000, voice: { warmth: 0.62, confidence: 0.8 } },
  { timestamp: 1759653600, voice: { warmth: 0.64, confidence: 0.78 } },
  { timestamp: 1759657200, voice: { warmth: 0.61, confidence: 0.82 } },
  { timestamp: 1759740000, voice: { warmth: 0.7, confidence: 0.75 } },
]

describe('PersonalityTrendChart', () => {
  beforeEach(() => {
    // jsdom has no ResizeObserver; recharts' ResponsiveContainer needs it.
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    )
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('renders the Label-role eyebrow title', () => {
    render(
      <PersonalityTrendChart
        title="Voice Over Time"
        history={HISTORY}
        field="voice"
        keys={['warmth', 'confidence']}
        labels={LABELS}
      />,
    )
    const title = screen.getByText('Voice Over Time')
    expect(title.className).toContain('text-xs')
    expect(title.className).toContain('font-medium')
    expect(title.className).toContain('uppercase')
    expect(title.className).toContain('tracking-wider')
  })

  it('shows the interaction-count caption', () => {
    render(
      <PersonalityTrendChart
        title="Voice Over Time"
        history={HISTORY}
        field="voice"
        keys={['warmth', 'confidence']}
        labels={LABELS}
      />,
    )
    expect(screen.getByText('Across 4 interactions')).toBeInTheDocument()
  })

  it('reads out the current value for every series in the inline legend', () => {
    render(
      <PersonalityTrendChart
        title="Voice Over Time"
        history={HISTORY}
        field="voice"
        keys={['warmth', 'confidence']}
        labels={LABELS}
      />,
    )
    // latest sorted point: warmth 0.7, confidence 0.75
    expect(screen.getByText('Warmth')).toBeInTheDocument()
    expect(screen.getByText('70%')).toBeInTheDocument()
    expect(screen.getByText('Confidence')).toBeInTheDocument()
    expect(screen.getByText('75%')).toBeInTheDocument()
  })

  it('colors series with chart tokens and contains no hex colors', () => {
    const { container } = render(
      <PersonalityTrendChart
        title="Voice Over Time"
        history={HISTORY}
        field="voice"
        keys={['warmth', 'confidence']}
        labels={LABELS}
      />,
    )
    const html = container.innerHTML
    expect(html).toContain('rgb(var(--chart-5))')
    expect(html).toContain('rgb(var(--chart-4))')
    expect(html).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
  })

  it('handles a mixed-day history and missing maps without crashing', () => {
    const { container } = render(
      <PersonalityTrendChart
        title="Traits Over Time"
        history={[
          { timestamp: 1759650000, traits: { openness: 0.5 } },
          { timestamp: 1759740000 },
          { timestamp: 1759830000, traits: {} },
        ]}
        field="traits"
        keys={['openness']}
        labels={{ openness: 'Openness' }}
      />,
    )
    expect(screen.getByText('Traits Over Time')).toBeInTheDocument()
    expect(screen.getByText('Across 3 interactions')).toBeInTheDocument()
    expect(container.innerHTML).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
  })
})
