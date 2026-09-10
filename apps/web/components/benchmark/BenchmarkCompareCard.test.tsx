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

import { BenchmarkCompareCard } from './BenchmarkCompareCard'

afterEach(() => { cleanup() })

describe('BenchmarkCompareCard', () => {
  it('renders empty state', () => {
    render(<BenchmarkCompareCard results={[]} />)
    expect(screen.getByText('Model Comparison')).toBeTruthy()
    expect(screen.getByText('Run benchmarks on multiple models to compare.')).toBeTruthy()
  })

  it('renders comparison with results', () => {
    const results: [string, Record<string, unknown>][] = [
      ['gpt2', { model: 'gpt2', throughput_tokens_per_sec: 50, memory_mb: 512, inference_time_ms: 200, total_tokens: 1000 }],
      ['llama', { model: 'llama', throughput_tokens_per_sec: 80, memory_mb: 1024, inference_time_ms: 150, total_tokens: 2000 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText('Model Comparison')).toBeTruthy()
    expect(screen.getByText('2 models')).toBeTruthy()
  })

  it('marks best values with star', () => {
    const results: [string, Record<string, unknown>][] = [
      ['slow', { throughput_tokens_per_sec: 30 }],
      ['fast', { throughput_tokens_per_sec: 100 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText(/fast ★/)).toBeTruthy()
  })

  it('shows model summary cards', () => {
    const results: [string, Record<string, unknown>][] = [
      ['gpt2', { total_tokens: 1000, inference_count: 5 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText('gpt2')).toBeTruthy()
    expect(screen.getByText('1000 tokens · 5 runs')).toBeTruthy()
  })

  it('hides metrics with no values', () => {
    const results: [string, Record<string, unknown>][] = [
      ['gpt2', { throughput_tokens_per_sec: 50 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText('Throughput')).toBeTruthy()
    expect(screen.queryByText('Memory')).toBeNull()
  })
})
