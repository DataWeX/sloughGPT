import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { TrafficCard, formatTokens } from './TrafficCard'

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
} as any

describe('formatTokens', () => {
  it('renders plain numbers below 1K', () => {
    expect(formatTokens(500)).toBe('500')
  })

  it('renders thousands as K', () => {
    expect(formatTokens(1500)).toBe('1.5K')
  })

  it('renders millions as M', () => {
    expect(formatTokens(2500000)).toBe('2.5M')
  })
})

describe('TrafficCard', () => {
  afterEach(cleanup)

  it('renders nothing when liveHealth is null', () => {
    const { container } = render(<TrafficCard liveHealth={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when there is no traffic yet', () => {
    const { container } = render(<TrafficCard liveHealth={{ ...base, requests_per_minute: 0, total_tokens: 0, avg_tokens_per_request: 0 }} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders requests/min, total tokens, and avg tokens/req', () => {
    render(<TrafficCard liveHealth={base} />)
    expect(screen.getByText('4.5')).toBeDefined()
    expect(screen.getByText('12.0K')).toBeDefined()
    expect(screen.getByText('160')).toBeDefined()
  })
})
