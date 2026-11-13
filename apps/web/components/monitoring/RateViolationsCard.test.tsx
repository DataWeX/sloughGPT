import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { RateViolationsCard } from './RateViolationsCard'

const base = {
  model_loaded: true,
  model_loading: false,
  model_type: 'gpt2',
  soul: null,
  is_inferencing: false,
  inference_count: 5,
  uptime_seconds: 120,
  request_count: 10,
  error_count: 2,
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
  model_events: [],
  rate_violations: [
    { path: '/inference/generate', count: 32, limit: 10, ts: 1723000000 },
    { path: '/chat/stream', count: 11, limit: 5, ts: 1722999900 },
  ],
  path_latencies: [],
  recent_errors: [],
} as any

describe('RateViolationsCard', () => {
  afterEach(cleanup)

  it('renders nothing when liveHealth is null', () => {
    const { container } = render(<RateViolationsCard liveHealth={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when there are no violations', () => {
    const { container } = render(<RateViolationsCard liveHealth={{ ...base, rate_violations: [] }} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders path and count/limit per violation', () => {
    render(<RateViolationsCard liveHealth={base} />)
    expect(screen.getByText(/\/inference\/generate/)).toBeDefined()
    expect(screen.getByText('32/10/s')).toBeDefined()
    expect(screen.getByText(/\/chat\/stream/)).toBeDefined()
    expect(screen.getByText('11/5/s')).toBeDefined()
  })

  it('renders a relative timestamp', () => {
    vi.setSystemTime(1723000000 * 1000)
    render(<RateViolationsCard liveHealth={base} />)
    expect(screen.getAllByText('just now').length).toBeGreaterThanOrEqual(1)
    vi.useRealTimers()
  })

  it('renders multiple violations', () => {
    render(<RateViolationsCard liveHealth={base} />)
    expect(screen.getByText(/\/inference\/generate/)).toBeDefined()
    expect(screen.getByText(/\/chat\/stream/)).toBeDefined()
  })

  it('renders count/limit ratios', () => {
    render(<RateViolationsCard liveHealth={base} />)
    expect(screen.getByText('32/10/s')).toBeDefined()
    expect(screen.getByText('11/5/s')).toBeDefined()
  })
})
