// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { BenchmarkHistoryCard } from './BenchmarkHistoryCard'

afterEach(() => { cleanup() })

describe('BenchmarkHistoryCard', () => {
  it('renders empty state', () => {
    render(<BenchmarkHistoryCard history={[]} />)
    expect(screen.getByText('Benchmark History')).toBeTruthy()
    expect(screen.getByText('No benchmark history yet.')).toBeTruthy()
  })

  it('renders stats summary', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50, latency: 200, memory: 512 },
      { timestamp: '2024-01-01T11:00:00Z', model: 'gpt2', throughput: 60, latency: 180, memory: 512 },
    ]
    render(<BenchmarkHistoryCard history={history} />)
    expect(screen.getByText('Benchmark History')).toBeTruthy()
    expect(screen.getByText('2')).toBeTruthy()
    expect(screen.getByText('55.0 tok/s')).toBeTruthy()
  })

  it('renders table with entries', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
    ]
    render(<BenchmarkHistoryCard history={history} />)
    const cells = screen.getAllByText('gpt2')
    expect(cells.length).toBeGreaterThanOrEqual(1)
  })

  it('filters by model', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
      { timestamp: '2024-01-01T11:00:00Z', model: 'llama', throughput: 80 },
    ]
    render(<BenchmarkHistoryCard history={history} />)
    const select = screen.getByLabelText('Filter by model')
    fireEvent.change(select, { target: { value: 'gpt2' } })
    const rows = document.querySelectorAll('tbody tr')
    expect(rows.length).toBe(1)
    expect(rows[0].textContent).toContain('gpt2')
  })

  it('sorts by clicking column headers', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
      { timestamp: '2024-01-01T11:00:00Z', model: 'llama', throughput: 80 },
    ]
    render(<BenchmarkHistoryCard history={history} />)
    const throughputHeader = screen.getByText('Throughput')
    fireEvent.click(throughputHeader)
    expect(screen.getByText('↓')).toBeTruthy()
  })

  it('calls onClear when clear button clicked', () => {
    const onClear = vi.fn()
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
    ]
    render(<BenchmarkHistoryCard history={history} onClear={onClear} />)
    fireEvent.click(screen.getByText('Clear'))
    expect(onClear).toHaveBeenCalled()
  })
})
