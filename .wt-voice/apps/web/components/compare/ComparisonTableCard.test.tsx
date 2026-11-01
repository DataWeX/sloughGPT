import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import ComparisonTableCard from './ComparisonTableCard'

describe('ComparisonTableCard', () => {
  afterEach(cleanup)

  const result = (tps: number, latency: number) => ({
    model: 'gpt2', perplexity: 15, latency_ms: latency, throughput: tps,
    num_parameters: 124_000_000, memory_mb: 500, throughput_tokens_per_sec: tps,
    inference_time_ms: latency, latency_p50_ms: latency, latency_p95_ms: latency, latency_p99_ms: latency,
  })

  const models = [{ id: 'gpt2', name: 'GPT-2' }, { id: 'qwen', name: 'Qwen' }]

  it('returns null with no results', () => {
    const { container } = render(<ComparisonTableCard completedResults={[]} models={models} bestMetrics={{}} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders model names and metrics', () => {
    const results: [string, any][] = [['gpt2', result(10, 200)], ['qwen', result(20, 100)]]
    render(<ComparisonTableCard completedResults={results} models={models} bestMetrics={{}} />)
    expect(screen.getByText('GPT-2')).toBeDefined()
    expect(screen.getByText('Qwen')).toBeDefined()
  })

  it('sorts by throughput descending', () => {
    const results: [string, any][] = [['gpt2', result(5, 200)], ['qwen', result(20, 100)]]
    const { container } = render(<ComparisonTableCard completedResults={results} models={models} bestMetrics={{}} />)
    const rows = container.querySelectorAll('tbody tr')
    expect(rows[0].textContent).toContain('Qwen')
    expect(rows[1].textContent).toContain('GPT-2')
  })

  it('shows best metric with dot indicator', () => {
    const results: [string, any][] = [['gpt2', result(10, 200)]]
    render(<ComparisonTableCard completedResults={results} models={models} bestMetrics={{ throughput: 10, latency: 200 }} />)
    const greenDots = document.querySelectorAll('.text-success')
    expect(greenDots.length).toBeGreaterThanOrEqual(2)
  })

  it('handles single result', () => {
    const results: [string, any][] = [['gpt2', result(10, 200)]]
    const { container } = render(<ComparisonTableCard completedResults={results} models={models} bestMetrics={{}} />)
    const rows = container.querySelectorAll('tbody tr')
    expect(rows.length).toBe(1)
  })

  it('handles three or more results', () => {
    const results: [string, any][] = [
      ['gpt2', result(5, 200)],
      ['qwen', result(20, 100)],
      ['llama', result(15, 150)],
    ]
    const { container } = render(<ComparisonTableCard completedResults={results} models={models} bestMetrics={{}} />)
    const rows = container.querySelectorAll('tbody tr')
    expect(rows.length).toBe(3)
  })

  it('shows table with rows', () => {
    const results: [string, any][] = [['gpt2', result(10, 200)]]
    const { container } = render(<ComparisonTableCard completedResults={results} models={models} bestMetrics={{}} />)
    expect(container.querySelector('tbody')).toBeDefined()
    expect(container.querySelectorAll('tbody tr').length).toBe(1)
  })

  it('handles empty bestMetrics', () => {
    const results: [string, any][] = [['gpt2', result(10, 200)]]
    const { container } = render(<ComparisonTableCard completedResults={results} models={models} bestMetrics={{}} />)
    expect(container.querySelectorAll('tbody tr').length).toBe(1)
  })
})
