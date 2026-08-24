import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { DiagnosticsCard } from './DiagnosticsCard'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

const healthy: LiveHealthSnapshot = {
  model_loaded: true,
  model_loading: false,
  model_type: 'qwen',
  device: 'cpu',
  soul: null,
  is_inferencing: false,
  inference_count: 3,
  uptime_seconds: 100,
  request_count: 5,
  error_count: 0,
  tokens_per_sec: 2.5,
  avg_latency_ms: 40,
  p95_latency_ms: 60,
  requests_per_minute: 1.5,
  total_tokens: 500,
  avg_tokens_per_request: 120,
  cpu_percent: 20,
  memory_percent: 40,
  health_score: 92,
  health_status: 'healthy',
  health_summary: 'All systems healthy',
  diagnoses: [
    { check: 'errors', severity: 'ok', score: 95, message: 'No errors in the last hour' },
    { check: 'latency', severity: 'ok', score: 90, message: 'Latency within bounds' },
  ],
  num_parameters: null,
  quantization: null,
  training_pool: null,
  model_metrics: [],
  model_events: [],
  rate_violations: [],
  health_history: [],
  memory_history: [],
  path_latencies: [],
  recent_errors: [],
}

describe('DiagnosticsCard', () => {
  afterEach(cleanup)

  it('renders empty state when liveHealth is null', () => {
    render(<DiagnosticsCard liveHealth={null} />)
    expect(screen.getAllByText('Diagnostics unavailable').length).toBeGreaterThanOrEqual(1)
  })

  it('renders empty state when there are no diagnoses and no summary', () => {
    render(<DiagnosticsCard liveHealth={{ ...healthy, diagnoses: [], health_summary: '' }} />)
    expect(screen.getAllByText('No diagnostics available').length).toBeGreaterThanOrEqual(1)
  })

  it('renders health score badge and summary', () => {
    render(<DiagnosticsCard liveHealth={healthy} />)
    expect(screen.getByText('92/100')).toBeDefined()
    expect(screen.getByText('All systems healthy')).toBeDefined()
  })

  it('renders each diagnosis with check name and message', () => {
    render(<DiagnosticsCard liveHealth={healthy} />)
    expect(screen.getByText('errors')).toBeDefined()
    expect(screen.getByText('No errors in the last hour')).toBeDefined()
    expect(screen.getByText('latency')).toBeDefined()
    expect(screen.getByText('Latency within bounds')).toBeDefined()
  })

  it('shows degraded badge for degraded status', () => {
    render(<DiagnosticsCard liveHealth={{ ...healthy, health_status: 'degraded', health_score: 65 }} />)
    expect(screen.getByText('65/100')).toBeDefined()
  })

  it('shows score even when status is unhealthy', () => {
    render(<DiagnosticsCard liveHealth={{ ...healthy, health_status: 'unhealthy', health_score: 30 }} />)
    expect(screen.getByText('30/100')).toBeDefined()
  })

  it('shows summary when diagnoses are empty', () => {
    render(<DiagnosticsCard liveHealth={{ ...healthy, diagnoses: [], health_summary: 'All good' }} />)
    expect(screen.getByText('All good')).toBeDefined()
  })
})
