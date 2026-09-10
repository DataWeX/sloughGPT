// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import { BenchmarkChartCard } from './BenchmarkChartCard'

afterEach(() => { cleanup() })

describe('BenchmarkChartCard', () => {
  it('renders empty state', () => {
    render(<BenchmarkChartCard history={[]} />)
    expect(screen.getByText('Metrics Over Time')).toBeTruthy()
    expect(screen.getByText('No benchmark history yet.')).toBeTruthy()
  })

  it('renders chart with data', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50, latency: 200, memory: 512 },
      { timestamp: '2024-01-01T11:00:00Z', model: 'gpt2', throughput: 55, latency: 180, memory: 512 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText('Metrics Over Time')).toBeTruthy()
    expect(screen.getByText(/2 runs/)).toBeTruthy()
  })

  it('shows multiple models', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
      { timestamp: '2024-01-01T11:00:00Z', model: 'llama', throughput: 80 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText(/2 models/)).toBeTruthy()
    expect(screen.getByText('gpt2')).toBeTruthy()
    expect(screen.getByText('llama')).toBeTruthy()
  })

  it('hides metrics with no data', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText('Throughput')).toBeTruthy()
    expect(screen.queryByText('Latency')).toBeNull()
    expect(screen.queryByText('Memory')).toBeNull()
  })

  it('renders recent runs section', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText('Recent runs')).toBeTruthy()
  })
})
