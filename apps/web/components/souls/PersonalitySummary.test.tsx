import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import PersonalitySummary, { deriveArchetype } from './PersonalitySummary'

const baseWeights = {
  personality: { warmth: 0.5, creativity: 0.5, empathy: 0.5, formality: 0.5, humor: 0.5, patience: 0.5, confidence: 0.5, curiosity: 0.5, directness: 0.5, optimism: 0.5 },
  cognition: { pattern_recognition: 0.5, long_context_handling: 0.5, abstract_reasoning: 0.5, factual_precision: 0.5 },
  emotion: { empathy_depth: 0.5, mood_responsiveness: 0.5, tone_flexibility: 0.5, sentiment_awareness: 0.5 },
}

describe('PersonalitySummary', () => {
  afterEach(cleanup)

  it('renders archetype badge when soul name is provided', () => {
    render(<PersonalitySummary traitWeights={baseWeights} currentSoulName="alice" />)
    expect(screen.getByText('alice')).toBeDefined()
  })

  it('renders archetype label in badge', () => {
    render(<PersonalitySummary traitWeights={baseWeights} currentSoulName="alice" />)
    expect(screen.getByText('The Balanced')).toBeDefined()
  })

  it('does not render archetype badge section when soul name is null', () => {
    const { container } = render(<PersonalitySummary traitWeights={baseWeights} currentSoulName={null} />)
    expect(container.querySelector('.border-b')).toBeNull()
  })

  it('renders 3 group stat cards with icons', () => {
    const { container } = render(<PersonalitySummary traitWeights={baseWeights} currentSoulName={null} />)
    const svgs = container.querySelectorAll('svg')
    expect(svgs.length).toBeGreaterThanOrEqual(3)
  })

  it('renders group labels in stat cards', () => {
    const { container } = render(<PersonalitySummary traitWeights={baseWeights} currentSoulName={null} />)
    // 'Personality' appears in both stat cards and individual traits section
    const labels = container.querySelectorAll('.uppercase.tracking-wider')
    const found: string[] = []
    labels.forEach(el => found.push(el.textContent || ''))
    expect(found).toContain('Personality')
    expect(found).toContain('Cognition')
    expect(found).toContain('Emotion')
  })

  it('renders individual trait names', () => {
    render(<PersonalitySummary traitWeights={baseWeights} currentSoulName={null} />)
    expect(screen.getByText('warmth')).toBeDefined()
    expect(screen.getByText('creativity')).toBeDefined()
    expect(screen.getByText('abstract reasoning')).toBeDefined()
  })

  it('renders individual trait percentages', () => {
    render(<PersonalitySummary traitWeights={baseWeights} currentSoulName={null} />)
    const all50 = screen.getAllByText('50')
    expect(all50.length).toBeGreaterThanOrEqual(2)
  })

  it('renders overall score when soul name is provided', () => {
    const { container } = render(<PersonalitySummary traitWeights={baseWeights} currentSoulName="alice" />)
    const overall = container.querySelector('.text-2xl.font-bold.text-destructive')
    expect(overall?.textContent).toBe('50')
  })

  it('shows trait count per group', () => {
    render(<PersonalitySummary traitWeights={baseWeights} currentSoulName={null} />)
    // Cognition and emotion both have 4 traits
    const count4 = screen.getAllByText(/· 4 traits/)
    expect(count4.length).toBe(2)
    expect(screen.getByText(/· 10 traits/)).toBeDefined()
  })

  it('renders stat cards even with empty groups', () => {
    const { container } = render(<PersonalitySummary traitWeights={{ personality: {}, cognition: {}, emotion: {} }} currentSoulName={null} />)
    // Stat cards still render (3 groups with 0 averages)
    const cards = container.querySelectorAll('.stat-card')
    expect(cards.length).toBe(3)
  })

  it('renders no individual traits when groups are empty', () => {
    const { container } = render(<PersonalitySummary traitWeights={{ personality: {}, cognition: {}, emotion: {} }} currentSoulName={null} />)
    // No individual trait names
    expect(container.querySelector('.grid.grid-cols-2')?.querySelectorAll('.capitalize').length || 0).toBe(0)
  })

  it('handles missing group gracefully', () => {
    render(<PersonalitySummary traitWeights={{ personality: { warmth: 0.8 } } as any} currentSoulName={null} />)
    expect(screen.getByText('warmth')).toBeDefined()
  })

  it('renders overall score 0 when no traits', () => {
    const { container } = render(<PersonalitySummary traitWeights={{ personality: {}, cognition: {}, emotion: {} }} currentSoulName="test" />)
    const overalls = container.querySelectorAll('.text-2xl')
    const overall = Array.from(overalls).find(el => el.textContent === '0')
    expect(overall).toBeDefined()
  })
})
