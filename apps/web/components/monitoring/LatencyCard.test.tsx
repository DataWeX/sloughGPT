import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { LatencyCard } from './LatencyCard'

const history = [
  { time: 't1', cpu: 10, mem: 20, latency: 100 },
  { time: 't2', cpu: 10, mem: 20, latency: 200 },
  { time: 't3', cpu: 10, mem: 20, latency: 300 },
  { time: 't4', cpu: 10, mem: 20, latency: 0 },
  { time: 't5', cpu: 10, mem: 20 },
]

describe('LatencyCard', () => {
  afterEach(cleanup)

  it('renders nothing when there are no positive latencies', () => {
    const { container } = render(<LatencyCard chartHistory={[{ time: 't1', cpu: 10, mem: 20 }]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when history is empty', () => {
    const { container } = render(<LatencyCard chartHistory={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('filters out zero and missing latencies', () => {
    render(<LatencyCard chartHistory={history} />)
    expect(screen.getByText('Avg')).toBeDefined()
  })

  it('computes avg latency from positive values', () => {
    render(<LatencyCard chartHistory={history} />)
    expect(screen.getByText('200ms')).toBeDefined()
  })

  it('computes min latency', () => {
    render(<LatencyCard chartHistory={history} />)
    expect(screen.getAllByText('100ms').length).toBeGreaterThanOrEqual(1)
  })

  it('computes max latency', () => {
    render(<LatencyCard chartHistory={history} />)
    expect(screen.getAllByText('300ms').length).toBeGreaterThanOrEqual(1)
  })

  it('computes p95 latency from a large set', () => {
    const many = Array.from({ length: 25 }, (_, i) => ({
      time: `t${i}`,
      cpu: 10,
      mem: 20,
      latency: 100 + i * 100,
    }))
    render(<LatencyCard chartHistory={many} />)
    expect(screen.getByText('2400ms')).toBeDefined()
    expect(screen.getByText('2500ms')).toBeDefined()
  })
})
