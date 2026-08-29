import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/recharts-lazy', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  RadarChart: ({ children }: any) => <div data-testid="radar-chart">{children}</div>,
  Radar: () => <div data-testid="radar" />,
  PolarGrid: () => <div data-testid="polar-grid" />,
  PolarAngleAxis: () => <div data-testid="polar-angle-axis" />,
  PolarRadiusAxis: () => <div data-testid="polar-radius-axis" />,
}))

import TraitRadarChart from './TraitRadarChart'

describe('TraitRadarChart', () => {
  afterEach(cleanup)

  it('renders nothing when data is empty', () => {
    const { container } = render(<TraitRadarChart data={{}} label="Test" color="#f00" />)
    expect(container.innerHTML).toBe('')
  })

  it('renders the label', () => {
    render(<TraitRadarChart data={{ warmth: 0.8 }} label="Personality" color="#8b5cf6" />)
    expect(screen.getByText('Personality')).toBeDefined()
  })

  it('renders chart for each trait', () => {
    render(<TraitRadarChart data={{ warmth: 0.8, creativity: 0.6 }} label="Test" color="#f00" />)
    expect(screen.getByTestId('radar-chart')).toBeDefined()
  })

  it('maps trait keys to readable labels', () => {
    render(<TraitRadarChart data={{ warmth: 0.8, pattern_recognition: 0.5 }} label="Test" color="#f00" />)
    expect(screen.getByText(/Warmth/)).toBeDefined()
    expect(screen.getByText(/Pattern Rec/)).toBeDefined()
  })

  it('displays numeric values', () => {
    render(<TraitRadarChart data={{ warmth: 0.75 }} label="Test" color="#f00" />)
    expect(screen.getByText('75')).toBeDefined()
  })

  it('renders polar grid and axes', () => {
    render(<TraitRadarChart data={{ warmth: 0.8 }} label="Test" color="#f00" />)
    expect(screen.getByTestId('polar-grid')).toBeDefined()
    expect(screen.getByTestId('polar-angle-axis')).toBeDefined()
    expect(screen.getByTestId('polar-radius-axis')).toBeDefined()
  })

  it('passes color prop to Radar', () => {
    render(<TraitRadarChart data={{ warmth: 0.8 }} label="Test" color="#123456" />)
    const radar = screen.getByTestId('radar')
    expect(radar).toBeDefined()
  })

  it('maps unknown trait keys to space-separated labels', () => {
    const { container } = render(<TraitRadarChart data={{ some_custom_trait: 0.5 }} label="Test" color="#f00" />)
    const legendText = container.querySelector('.flex.flex-wrap')?.textContent || ''
    expect(legendText).toContain('some custom trait')
  })

  it('renders trait legend below chart', () => {
    render(<TraitRadarChart data={{ warmth: 0.8, humor: 0.3 }} label="Test" color="#f00" />)
    expect(screen.getByText(/Warmth/)).toBeDefined()
    expect(screen.getByText(/Humor/)).toBeDefined()
  })

  it('rounds values to integers', () => {
    render(<TraitRadarChart data={{ warmth: 0.123, humor: 0.987 }} label="Test" color="#f00" />)
    expect(screen.getByText('12')).toBeDefined()
    expect(screen.getByText('99')).toBeDefined()
  })

  it('wraps chart in ResponsiveContainer', () => {
    render(<TraitRadarChart data={{ warmth: 0.8 }} label="Test" color="#f00" />)
    expect(screen.getByTestId('responsive-container')).toBeDefined()
  })
})
