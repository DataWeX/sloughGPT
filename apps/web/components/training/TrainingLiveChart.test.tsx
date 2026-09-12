// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('recharts', () => ({
  LineChart: ({ children, ...props }: any) => <div data-testid="line-chart" {...props}>{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  CartesianGrid: () => null,
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  StatCard: ({ label, value }: any) => <div data-testid="stat-card">{label}: {value}</div>,
  KpiGrid: ({ children }: any) => <div data-testid="kpi-grid">{children}</div>,
}))

import { TrainingLiveChart } from './TrainingLiveChart'

afterEach(() => { cleanup() })

const defaultProps = {
  lossHistory: [],
  progress: 0,
  epoch: 0,
  totalEpochs: 0,
  globalStep: 0,
  totalSteps: 0,
  stepsPerSec: null,
  eta: null,
}

describe('TrainingLiveChart', () => {
  it('renders without crashing', () => {
    render(<TrainingLiveChart {...defaultProps} />)
    expect(screen.getByText('Training Progress')).toBeDefined()
  })

  it('shows waiting message when no loss data', () => {
    render(<TrainingLiveChart {...defaultProps} />)
    expect(screen.getByText('Waiting for loss data...')).toBeDefined()
  })

  it('renders KPI grid with correct stats', () => {
    render(
      <TrainingLiveChart
        {...defaultProps}
        progress={0.5}
        epoch={3}
        totalEpochs={10}
        globalStep={500}
        totalSteps={1000}
      />
    )
    expect(screen.getByText('50.0%')).toBeDefined()
    expect(screen.getByText('3/10')).toBeDefined()
    expect(screen.getByText('500/1000')).toBeDefined()
  })

  it('shows chart when loss data has more than 1 point', () => {
    const lossHistory = [
      { step: 1, loss: 1.0 },
      { step: 2, loss: 0.8 },
    ]
    render(<TrainingLiveChart {...defaultProps} lossHistory={lossHistory} />)
    expect(screen.queryByText('Waiting for loss data...')).toBeNull()
    expect(screen.getByTestId('line-chart')).toBeDefined()
  })

  it('shows steps per sec when provided', () => {
    render(<TrainingLiveChart {...defaultProps} stepsPerSec={2.5} />)
    expect(screen.getByText('2.5 steps/s')).toBeDefined()
  })

  it('formats ETA in minutes and seconds', () => {
    render(<TrainingLiveChart {...defaultProps} eta={125} />)
    expect(screen.getByText('2m 5s')).toBeDefined()
  })

  it('formats ETA as dash when null', () => {
    render(<TrainingLiveChart {...defaultProps} eta={null} />)
    expect(screen.getByText('—')).toBeDefined()
  })
})
