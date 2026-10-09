import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { QualityCard } from './QualityCard'

const quality = {
  status: 'ok',
  total_responses: 12,
  coherence_score: 0.85,
  quality_score: 0.9,
  repetition_rate: 0.05,
  avg_length: 42.3,
  empty_rate: 0.01,
}

describe('QualityCard', () => {
  afterEach(cleanup)

  it('renders nothing when total responses is 0', () => {
    const { container } = render(<QualityCard quality={{ ...quality, total_responses: 0 } as any} stats={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders coherence and quality scores', () => {
    render(<QualityCard quality={quality as any} stats={null} />)
    expect(screen.getByText('0.85')).toBeDefined()
    expect(screen.getByText('0.90')).toBeDefined()
  })

  it('renders total responses', () => {
    render(<QualityCard quality={quality as any} stats={null} />)
    expect(screen.getByText('12')).toBeDefined()
  })

  it('renders repetition rate as a percentage', () => {
    render(<QualityCard quality={quality as any} stats={null} />)
    expect(screen.getByText('5.0%')).toBeDefined()
  })

  it('renders average length and empty rate', () => {
    render(<QualityCard quality={quality as any} stats={null} />)
    expect(screen.getByText('Avg: 42.3w')).toBeDefined()
    expect(screen.getByText('Empty: 1.0%')).toBeDefined()
  })

  it('renders token stats when provided', () => {
    render(<QualityCard quality={quality as any} stats={{ total: 100, avg_tokens: 150, models: ['gpt2'] }} />)
    expect(screen.getByText('Tokens: 150')).toBeDefined()
  })
})
