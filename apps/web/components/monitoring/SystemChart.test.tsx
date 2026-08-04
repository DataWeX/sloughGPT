import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  CartesianGrid: () => null,
  Legend: () => null,
}))

import { SystemChart } from '@/components/monitoring/SystemChart'

describe('SystemChart', () => {
  it('renders chart container', () => {
    render(<SystemChart data={[{ time: '10:00', cpu: 45, mem: 60 }]} />)
    expect(screen.getAllByTestId('responsive-container').length).toBeGreaterThanOrEqual(1)
  })

  it('renders line chart', () => {
    render(<SystemChart data={[{ time: '10:00', cpu: 45, mem: 60 }]} />)
    expect(screen.getAllByTestId('line-chart').length).toBeGreaterThanOrEqual(1)
  })

  it('renders with multiple data points', () => {
    render(
      <SystemChart data={[
        { time: '10:00', cpu: 45, mem: 60 },
        { time: '10:01', cpu: 50, mem: 62 },
      ]} />,
    )
    expect(screen.getAllByTestId('line-chart').length).toBeGreaterThanOrEqual(1)
  })

  it('renders with empty data', () => {
    render(<SystemChart data={[]} />)
    expect(screen.getAllByTestId('responsive-container').length).toBeGreaterThanOrEqual(1)
  })
})
