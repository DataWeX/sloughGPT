import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { PathLatenciesCard } from './PathLatenciesCard'

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
  model_metrics: [],
  path_latencies: [
    { path: '/inference/generate', avg_ms: 320.5, count: 18, p95_ms: 610.2 },
    { path: '/chat/stream', avg_ms: 150.0, count: 7, p95_ms: 210.4 },
  ],
} as any

describe('PathLatenciesCard', () => {
  afterEach(cleanup)

  it('renders nothing when liveHealth is null', () => {
    const { container } = render(<PathLatenciesCard liveHealth={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when there are no path latencies', () => {
    const { container } = render(<PathLatenciesCard liveHealth={{ ...base, path_latencies: [] }} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders endpoint paths with avg and p95', () => {
    render(<PathLatenciesCard liveHealth={base} />)
    expect(screen.getByText('/inference/generate')).toBeDefined()
    expect(screen.getByText('avg 320.5ms')).toBeDefined()
    expect(screen.getByText('p95 610.2ms')).toBeDefined()
  })

  it('renders request counts per path', () => {
    render(<PathLatenciesCard liveHealth={base} />)
    expect(screen.getByText('×18')).toBeDefined()
    expect(screen.getByText('×7')).toBeDefined()
  })
})
