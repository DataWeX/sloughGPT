import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/lib/recharts-lazy', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  RadarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="radar-chart">{children}</div>,
  Bar: () => null,
  Radar: () => null,
  PolarGrid: () => null,
  PolarAngleAxis: () => null,
  PolarRadiusAxis: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  CartesianGrid: () => null,
  Legend: () => null,
}))

import VisualComparisonCard from '@/components/compare/VisualComparisonCard'

describe('VisualComparisonCard', () => {
  it('returns null when < 2 data points', () => {
    const { container } = render(<VisualComparisonCard chartData={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('returns null with single data point', () => {
    const { container } = render(
      <VisualComparisonCard chartData={[{ name: 'A', throughput: 10, latency: 50, memory: 100 }]} />,
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders when >= 2 data points', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'A', throughput: 10, latency: 50, memory: 100 },
        { name: 'B', throughput: 20, latency: 30, memory: 80 },
      ]} />,
    )
    expect(screen.getAllByText('Visual Comparison').length).toBeGreaterThanOrEqual(1)
  })

  it('renders throughput and latency chart labels', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'X', throughput: 5, latency: 100, memory: 200 },
        { name: 'Y', throughput: 8, latency: 80, memory: 150 },
      ]} />,
    )
    expect(screen.getAllByText(/Throughput/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/latency/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders bar charts', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'A', throughput: 10, latency: 50, memory: 100 },
        { name: 'B', throughput: 20, latency: 30, memory: 80 },
      ]} />,
    )
    expect(screen.getAllByTestId('bar-chart').length).toBeGreaterThanOrEqual(2)
  })

  it('passes chartData with name field to recharts', () => {
    const data = [
      { name: 'GPT-2', throughput: 10, latency: 50, memory: 100 },
      { name: 'Qwen', throughput: 20, latency: 30, memory: 80 },
    ]
    render(<VisualComparisonCard chartData={data} />)
    expect(screen.getAllByTestId('bar-chart').length).toBeGreaterThanOrEqual(1)
  })

  it('renders responsive container', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'A', throughput: 10, latency: 50, memory: 100 },
        { name: 'B', throughput: 20, latency: 30, memory: 80 },
      ]} />,
    )
    expect(screen.getAllByTestId('responsive-container').length).toBeGreaterThanOrEqual(1)
  })
})
