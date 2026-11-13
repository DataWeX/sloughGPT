// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { SoulPersonalityCard } from './SoulPersonalityCard'

afterEach(() => { cleanup() })

const personality = {
  warmth: 0.9,
  creativity: 0.7,
  curiosity: 0.8,
  confidence: 0.6,
}

describe('SoulPersonalityCard', () => {
  it('returns null for empty personality', () => {
    const { container } = render(<SoulPersonalityCard personality={{}} />)
    expect(container.querySelector('[data-testid="soul-personality"]')).toBeNull()
  })

  it('returns null for undefined personality', () => {
    const { container } = render(<SoulPersonalityCard personality={undefined} />)
    expect(container.querySelector('[data-testid="soul-personality"]')).toBeNull()
  })

  it('renders personality card', () => {
    render(<SoulPersonalityCard personality={personality} />)
    expect(screen.getAllByTestId('soul-personality').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Personality').length).toBeGreaterThanOrEqual(1)
  })

  it('shows soul name', () => {
    render(<SoulPersonalityCard personality={personality} soulName="Friendly" />)
    expect(screen.getAllByText(/Friendly/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows trait labels', () => {
    render(<SoulPersonalityCard personality={personality} />)
    expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Creativity').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Curiosity').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Confidence').length).toBeGreaterThanOrEqual(1)
  })

  it('shows percentage values', () => {
    render(<SoulPersonalityCard personality={personality} />)
    expect(screen.getAllByText('90%').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('70%').length).toBeGreaterThanOrEqual(1)
  })

  it('shows average score', () => {
    render(<SoulPersonalityCard personality={personality} />)
    expect(screen.getAllByText(/avg/).length).toBeGreaterThanOrEqual(1)
    const { container } = render(<SoulPersonalityCard personality={personality} />)
    expect(container.textContent).toContain('75%')
  })

  it('shows traits badges when provided', () => {
    render(<SoulPersonalityCard personality={personality} traits={['friendly', 'helpful']} />)
    expect(screen.getAllByText('friendly').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('helpful').length).toBeGreaterThanOrEqual(1)
  })

  it('renders bar widths', () => {
    const { container } = render(<SoulPersonalityCard personality={personality} />)
    const bars = container.querySelectorAll('[style*="width"]')
    expect(bars.length).toBeGreaterThan(0)
  })
})
