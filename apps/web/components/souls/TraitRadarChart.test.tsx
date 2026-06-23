// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('recharts', () => ({
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
})
