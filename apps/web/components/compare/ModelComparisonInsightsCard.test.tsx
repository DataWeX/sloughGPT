// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ModelComparisonInsightsCard } from './ModelComparisonInsightsCard'

afterEach(() => { cleanup() })

const models = [
  { id: 'gpt2', name: 'GPT-2' },
  { id: 'qwen', name: 'Qwen 0.5B' },
  { id: 'smol', name: 'SmolLM2' },
]

const results: [string, { throughput_tokens_per_sec: number; inference_time_ms: number; latency_p95_ms?: number; memory_mb: number; num_parameters: number; error?: string }][] = [
  ['gpt2', { throughput_tokens_per_sec: 5.2, inference_time_ms: 3000, memory_mb: 500, num_parameters: 124000000 }],
  ['qwen', { throughput_tokens_per_sec: 8.3, inference_time_ms: 2000, memory_mb: 1000, num_parameters: 500000000 }],
  ['smol', { throughput_tokens_per_sec: 12.1, inference_time_ms: 1500, memory_mb: 300, num_parameters: 135000000 }],
]

const bestMetrics = { throughput: 12.1, latency: 1500, p95: 1800, params: 500000000 }

describe('ModelComparisonInsightsCard', () => {
  it('returns null for less than 2 results', () => {
    const { container } = render(<ModelComparisonInsightsCard completedResults={[results[0]]} models={models} bestMetrics={bestMetrics} />)
    expect(container.querySelector('[data-testid="model-comparison-insights"]')).toBeNull()
  })

  it('renders card with 2+ results', () => {
    render(<ModelComparisonInsightsCard completedResults={results} models={models} bestMetrics={bestMetrics} />)
    expect(screen.getAllByTestId('model-comparison-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Comparison Insights').length).toBeGreaterThanOrEqual(1)
  })

  it('shows fastest model', () => {
    render(<ModelComparisonInsightsCard completedResults={results} models={models} bestMetrics={bestMetrics} />)
    expect(screen.getAllByText('SmolLM2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows highest throughput model', () => {
    render(<ModelComparisonInsightsCard completedResults={results} models={models} bestMetrics={bestMetrics} />)
    expect(screen.getAllByText('SmolLM2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows most efficient model', () => {
    render(<ModelComparisonInsightsCard completedResults={results} models={models} bestMetrics={bestMetrics} />)
    expect(screen.getAllByText('SmolLM2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows models compared count', () => {
    render(<ModelComparisonInsightsCard completedResults={results} models={models} bestMetrics={bestMetrics} />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('handles empty results', () => {
    const { container } = render(<ModelComparisonInsightsCard completedResults={[]} models={models} bestMetrics={bestMetrics} />)
    expect(container.querySelector('[data-testid="model-comparison-insights"]')).toBeNull()
  })

  it('skips error results', () => {
    const withError: typeof results = [
      ['gpt2', { throughput_tokens_per_sec: 5.2, inference_time_ms: 3000, memory_mb: 500, num_parameters: 124000000, error: 'timeout' }],
      ['qwen', { throughput_tokens_per_sec: 8.3, inference_time_ms: 2000, memory_mb: 1000, num_parameters: 500000000 }],
    ]
    render(<ModelComparisonInsightsCard completedResults={withError} models={models} bestMetrics={bestMetrics} />)
    expect(screen.getAllByTestId('model-comparison-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Qwen 0.5B').length).toBeGreaterThanOrEqual(1)
  })
})
