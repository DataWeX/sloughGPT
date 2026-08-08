import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ModelEventsCard } from './ModelEventsCard'

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
  model_events: [
    { type: 'load', model: 'gpt2', detail: 'loaded from cache', ts: 1723000000 },
    { type: 'error', model: 'gpt2', detail: 'generation failed', ts: 1722999900 },
    { type: 'unload', model: 'gpt2', detail: '', ts: 1722999800 },
  ],
  path_latencies: [],
  recent_errors: [],
} as any

describe('ModelEventsCard', () => {
  afterEach(cleanup)

  it('shows empty state when liveHealth is null', () => {
    render(<ModelEventsCard liveHealth={null} />)
    expect(screen.getByText('No model events yet')).toBeDefined()
  })

  it('shows empty state when there are no model events', () => {
    render(<ModelEventsCard liveHealth={{ ...base, model_events: [] }} />)
    expect(screen.getByText('No model events yet')).toBeDefined()
  })

  it('renders event type, model, and detail', () => {
    render(<ModelEventsCard liveHealth={base} />)
    expect(screen.getByText('load')).toBeDefined()
    expect(screen.getByText('error')).toBeDefined()
    expect(screen.getByText('unload')).toBeDefined()
    expect(screen.getAllByText('gpt2').length).toBeGreaterThanOrEqual(3)
    expect(screen.getByText('loaded from cache')).toBeDefined()
  })

  it('renders a relative timestamp', () => {
    vi.setSystemTime(1723000000 * 1000)
    render(<ModelEventsCard liveHealth={base} />)
    expect(screen.getAllByText('just now').length).toBeGreaterThanOrEqual(1)
    vi.useRealTimers()
  })
})
