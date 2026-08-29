import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { LatencyCard } from './LatencyCard'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

function makeHealth(overrides: Partial<LiveHealthSnapshot> = {}): LiveHealthSnapshot {
  return {
    model_loaded: true, model_loading: false, model_type: 'gpt2', device: 'cpu', soul: null,
    is_inferencing: false, inference_count: 10, uptime_seconds: 300, request_count: 10,
    error_count: 0, tokens_per_sec: 12, avg_latency_ms: 0, p95_latency_ms: 0,
    requests_per_minute: 5, total_tokens: 200, avg_tokens_per_request: 20,
    cpu_percent: 25, memory_percent: 60, health_score: 85, health_status: 'healthy',
    health_summary: 'OK', diagnoses: [], num_parameters: null, quantization: null,
    training_pool: null, model_metrics: [], model_events: [], rate_violations: [],
    health_history: [], memory_history: [], path_latencies: [], recent_errors: [],
    ...overrides,
  }
}

describe('LatencyCard', () => {
  afterEach(cleanup)

  it('renders nothing when avg and p95 are zero', () => {
    const { container } = render(<LatencyCard liveHealth={makeHealth({ avg_latency_ms: 0, p95_latency_ms: 0 })} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when liveHealth is null', () => {
    const { container } = render(<LatencyCard liveHealth={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders avg and p95 from backend data', () => {
    render(<LatencyCard liveHealth={makeHealth({ avg_latency_ms: 150, p95_latency_ms: 350 })} />)
    expect(screen.getByText('Avg')).toBeDefined()
    expect(screen.getByText('P95')).toBeDefined()
    expect(screen.getByText('150ms')).toBeDefined()
    expect(screen.getByText('350ms')).toBeDefined()
  })

  it('shows skeleton when only avg is provided', () => {
    render(<LatencyCard liveHealth={makeHealth({ avg_latency_ms: 100, p95_latency_ms: 0 })} />)
    expect(screen.getByText('100ms')).toBeDefined()
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(1)
  })
})
