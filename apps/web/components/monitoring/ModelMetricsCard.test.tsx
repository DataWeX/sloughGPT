import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ModelMetricsCard } from './ModelMetricsCard'

const base = {
  model_loaded: true,
  model_loading: false,
  model_type: 'gpt2',
  soul: null,
  is_inferencing: false,
  inference_count: 5,
  uptime_seconds: 120,
  request_count: 10,
  error_count: 0,
  tokens_per_sec: 12.5,
  avg_latency_ms: 80,
  requests_per_minute: 4.5,
  total_tokens: 12000,
  avg_tokens_per_request: 160,
  cpu_percent: 45,
  memory_percent: 60,
  health_score: 85,
  health_status: 'healthy',
  health_summary: '',
  diagnoses: [],
  num_parameters: null,
  quantization: null,
  training_pool: null,
  model_metrics: [
    { model: 'gpt2', count: 12, total_tokens: 9000, tokens_per_sec: 9.5, avg_tokens: 750 },
    { model: 'qwen-0.5b', count: 3, total_tokens: 1200, tokens_per_sec: 4.2, avg_tokens: 400 },
  ],
} as any

describe('ModelMetricsCard', () => {
  afterEach(cleanup)

  it('renders nothing when liveHealth is null', () => {
    const { container } = render(<ModelMetricsCard liveHealth={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when there are no model metrics', () => {
    const { container } = render(<ModelMetricsCard liveHealth={{ ...base, model_metrics: [] }} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders model names and per-model stats', () => {
    render(<ModelMetricsCard liveHealth={base} />)
    expect(screen.getByText('gpt2')).toBeDefined()
    expect(screen.getByText('qwen-0.5b')).toBeDefined()
    expect(screen.getByText('9.5 tok/s')).toBeDefined()
    expect(screen.getByText(/12 requests/)).toBeDefined()
  })

  it('formats large token totals', () => {
    render(<ModelMetricsCard liveHealth={base} />)
    expect(screen.getByText(/9.0K tokens/)).toBeDefined()
  })

  it('shows tokens per second for each model', () => {
    render(<ModelMetricsCard liveHealth={base} />)
    expect(screen.getByText('9.5 tok/s')).toBeDefined()
    expect(screen.getByText('4.2 tok/s')).toBeDefined()
  })

  it('shows request count for each model', () => {
    render(<ModelMetricsCard liveHealth={base} />)
    expect(screen.getByText(/12 requests/)).toBeDefined()
    expect(screen.getByText(/3 requests/)).toBeDefined()
  })

  it('renders per-model stats sections', () => {
    render(<ModelMetricsCard liveHealth={base} />)
    expect(screen.getByText('gpt2')).toBeDefined()
    expect(screen.getByText('qwen-0.5b')).toBeDefined()
    expect(screen.getByText('9.5 tok/s')).toBeDefined()
    expect(screen.getByText('4.2 tok/s')).toBeDefined()
  })

  it('renders multiple models', () => {
    render(<ModelMetricsCard liveHealth={base} />)
    expect(screen.getByText('gpt2')).toBeDefined()
    expect(screen.getByText('qwen-0.5b')).toBeDefined()
  })
})
