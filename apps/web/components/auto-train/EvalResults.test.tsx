/**
 * @vitest-environment jsdom
 */
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { EvalResults, type EvalData } from './EvalResults'

const baseData: EvalData = {
  baseline: { perplexity: 12.5, bleu: 0.35, tokens_per_sec: 45, personality_score: 0.72 },
  with_adapter: { perplexity: 10.2, bleu: 0.42, tokens_per_sec: 44, personality_score: 0.78 },
  delta: { verdict: 'improved', perplexity_delta: -2.3, bleu_delta: 0.07, throughput_delta: -1, personality_delta: 0.06 },
}

describe('EvalResults', () => {
  it('renders verdict badge for improved', () => {
    const { container } = render(<EvalResults data={baseData} />)
    expect(container.textContent).toContain('Rating: Improved')
  })

  it('renders verdict badge for degraded', () => {
    const degraded: EvalData = {
      baseline: { perplexity: 10, bleu: 0.5, tokens_per_sec: 50, personality_score: 0.8 },
      with_adapter: { perplexity: 15, bleu: 0.3, tokens_per_sec: 40, personality_score: 0.6 },
      delta: { verdict: 'degraded', perplexity_delta: 5, bleu_delta: -0.2, throughput_delta: -10, personality_delta: -0.2 },
    }
    const { container } = render(<EvalResults data={degraded} />)
    expect(container.textContent).toContain('Rating: Degraded')
  })

  it('renders verdict badge for unchanged', () => {
    const unchanged: EvalData = {
      baseline: { perplexity: 10, bleu: 0.5, tokens_per_sec: 50, personality_score: 0.8 },
      with_adapter: { perplexity: 10, bleu: 0.5, tokens_per_sec: 50, personality_score: 0.8 },
      delta: { verdict: 'unchanged', perplexity_delta: 0, bleu_delta: 0, throughput_delta: 0, personality_delta: 0 },
    }
    const { container } = render(<EvalResults data={unchanged} />)
    expect(container.textContent).toContain('Rating: Unchanged')
  })

  it('renders all four metric labels', () => {
    const { container } = render(<EvalResults data={baseData} />)
    expect(container.textContent).toContain('Perplexity')
    expect(container.textContent).toContain('BLEU')
    expect(container.textContent).toContain('Throughput')
    expect(container.textContent).toContain('Personality Score')
  })

  it('shows baseline and adapted values', () => {
    const { container } = render(<EvalResults data={baseData} />)
    expect(container.textContent).toContain('12.50')
    expect(container.textContent).toContain('10.20')
  })

  it('shows delta with sign for improved metrics', () => {
    const { container } = render(<EvalResults data={baseData} />)
    expect(container.textContent).toContain('-2.30')
    expect(container.textContent).toContain('+0.07')
  })

  it('shows report section when data has report', () => {
    const withReport: EvalData = {
      ...baseData,
      report: 'Detailed evaluation report',
    }
    const { container } = render(<EvalResults data={withReport} />)
    expect(container.textContent).toContain('View Report')
    expect(container.textContent).toContain('Detailed evaluation report')
  })

  it('hides report section when no report field', () => {
    const { container } = render(<EvalResults data={baseData} />)
    expect(container.textContent).not.toContain('View Report')
  })
})
