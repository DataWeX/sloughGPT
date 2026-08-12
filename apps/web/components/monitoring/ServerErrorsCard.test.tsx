import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ServerErrorsCard } from './ServerErrorsCard'

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
  path_latencies: [],
  recent_errors: [
    { path: '/inference/generate', method: 'POST', status: 500, message: 'generation timed out', error_type: 'TimeoutError', ts: 1723000000 },
    { path: '/chat', method: 'POST', status: 422, message: 'bad payload', error_type: 'ValidationError', ts: 1722999900 },
  ],
} as any

describe('ServerErrorsCard', () => {
  afterEach(cleanup)

  it('shows empty state when liveHealth is null', () => {
    render(<ServerErrorsCard liveHealth={null} />)
    expect(screen.getByText('No errors recorded yet')).toBeDefined()
  })

  it('shows empty state when there are no server errors', () => {
    render(<ServerErrorsCard liveHealth={{ ...base, recent_errors: [] }} />)
    expect(screen.getByText('No errors recorded yet')).toBeDefined()
  })

  it('renders method, path, status, and message per error', () => {
    render(<ServerErrorsCard liveHealth={base} />)
    expect(screen.getByText(/POST \/inference\/generate/)).toBeDefined()
    expect(screen.getByText('500')).toBeDefined()
    expect(screen.getByText(/TimeoutError/)).toBeDefined()
    expect(screen.getByText('generation timed out')).toBeDefined()
    expect(screen.getByText('422')).toBeDefined()
  })

  it('renders a relative timestamp', () => {
    vi.setSystemTime(1723000000 * 1000)
    render(<ServerErrorsCard liveHealth={base} />)
    expect(screen.getByText('just now')).toBeDefined()
    vi.useRealTimers()
  })

  it('renders multiple errors', () => {
    render(<ServerErrorsCard liveHealth={base} />)
    expect(screen.getByText(/POST \/inference\/generate/)).toBeDefined()
    expect(screen.getByText(/POST \/chat/)).toBeDefined()
  })

  it('renders error status codes', () => {
    render(<ServerErrorsCard liveHealth={base} />)
    expect(screen.getByText('500')).toBeDefined()
    expect(screen.getByText('422')).toBeDefined()
  })

  it('renders error messages', () => {
    render(<ServerErrorsCard liveHealth={base} />)
    expect(screen.getByText('generation timed out')).toBeDefined()
    expect(screen.getByText('bad payload')).toBeDefined()
  })

  it('renders error types', () => {
    render(<ServerErrorsCard liveHealth={base} />)
    expect(screen.getByText(/TimeoutError/)).toBeDefined()
    expect(screen.getByText(/ValidationError/)).toBeDefined()
  })
})
