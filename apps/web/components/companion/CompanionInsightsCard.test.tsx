// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { CompanionInsightsCard } from './CompanionInsightsCard'

afterEach(() => { cleanup() })

const traits = {
  name: 'Test Companion',
  warmth: 0.9,
  curiosity: 0.7,
  creativity: 0.5,
  confidence: 0.8,
  humor: 0.3,
}

const presets = [
  { id: 'warm', name: 'Warm', traits: { warmth: 0.9, curiosity: 0.5, creativity: 0.5, confidence: 0.5, humor: 0.5 } },
  { id: 'curious', name: 'Curious', traits: { warmth: 0.5, curiosity: 0.9, creativity: 0.5, confidence: 0.5, humor: 0.5 } },
]

describe('CompanionInsightsCard', () => {
  it('returns null for null traits', () => {
    const { container } = render(<CompanionInsightsCard traits={null} presets={[]} />)
    expect(container.querySelector('[data-testid="companion-insights"]')).toBeNull()
  })

  it('returns null for empty traits', () => {
    const { container } = render(<CompanionInsightsCard traits={{ name: 'empty', warmth: 0, curiosity: 0, creativity: 0, confidence: 0, humor: 0 }} presets={[]} />)
    expect(container.querySelector('[data-testid="companion-insights"]')).toBeNull()
  })

  it('renders card with traits', () => {
    render(<CompanionInsightsCard traits={traits} presets={presets} />)
    expect(screen.getAllByTestId('companion-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Personality Profile').length).toBeGreaterThanOrEqual(1)
  })

  it('shows dominant trait', () => {
    render(<CompanionInsightsCard traits={traits} presets={presets} />)
    expect(screen.getAllByText('Dominant').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('warmth').length).toBeGreaterThanOrEqual(1)
  })

  it('shows weakest trait', () => {
    render(<CompanionInsightsCard traits={traits} presets={presets} />)
    expect(screen.getAllByText('Weakest').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('humor').length).toBeGreaterThanOrEqual(1)
  })

  it('shows personality type', () => {
    render(<CompanionInsightsCard traits={traits} presets={presets} />)
    expect(screen.getAllByText('Strong warmth').length).toBeGreaterThanOrEqual(1)
  })

  it('shows trait badges with percentages', () => {
    render(<CompanionInsightsCard traits={traits} presets={presets} />)
    expect(screen.getAllByText(/warmth 90%/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/curiosity 70%/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows preset count', () => {
    render(<CompanionInsightsCard traits={traits} presets={presets} />)
    expect(screen.getAllByText(/2 presets available/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows balance level', () => {
    render(<CompanionInsightsCard traits={traits} presets={presets} />)
    expect(screen.getAllByText('Skewed').length).toBeGreaterThanOrEqual(1)
  })
})
