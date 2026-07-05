import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import SummaryCard from './SummaryCard'

describe('SummaryCard', () => {
  afterEach(cleanup)

  const models = [{ id: 'gpt2', name: 'GPT-2' }, { id: 'qwen', name: 'Qwen' }]
  const result = (tps: number) => ({
    model: 'gpt2', perplexity: 15, latency_ms: 100, throughput: tps,
    num_parameters: 124_000_000, memory_mb: 500, throughput_tokens_per_sec: tps,
    inference_time_ms: 100, latency_p50_ms: 100, latency_p95_ms: 100, latency_p99_ms: 100,
  })

  it('returns null with fewer than 2 results', () => {
    const { container } = render(<SummaryCard completedResults={[['gpt2', result(10)]]} models={models} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders throughput for each model', () => {
    render(<SummaryCard completedResults={[['gpt2', result(10)], ['qwen', result(20)]]} models={models} />)
    expect(screen.getByText('10.0 tok/s')).toBeDefined()
    expect(screen.getByText('20.0 tok/s')).toBeDefined()
  })

  it('shows model names as labels', () => {
    render(<SummaryCard completedResults={[['gpt2', result(10)], ['qwen', result(20)]]} models={models} />)
    expect(screen.getByText('GPT-2')).toBeDefined()
    expect(screen.getByText('Qwen')).toBeDefined()
  })

  it('falls back to model id when name is missing', () => {
    render(<SummaryCard completedResults={[['unknown', result(15)], ['gpt2', result(10)]]} models={[{ id: 'unknown', name: '' }, { id: 'gpt2', name: 'GPT-2' }]} />)
    expect(screen.getByText('unknown')).toBeDefined()
  })
})
