import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  ComposedChart: ({ children }: { children: React.ReactNode }) => <div data-testid="composed-chart">{children}</div>,
  Line: () => null,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  CartesianGrid: () => null,
  Legend: () => null,
}))

import { TrendChart } from './TrendChart'

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
  rate_violations: [],
  health_history: [
    { score: 90, status: 'healthy', ts: 1000 },
    { score: 85, status: 'healthy', ts: 1010 },
  ],
  memory_history: [
    { rss_mb: 300, virtual_mb: 400, system_percent: 55, ts: 1002 },
    { rss_mb: 320, virtual_mb: 420, system_percent: 58, ts: 1012 },
  ],
  path_latencies: [],
  recent_errors: [],
} as any

describe('TrendChart', () => {
  afterEach(cleanup)

  it('renders nothing meaningful when liveHealth is null (empty-state text)', () => {
    render(<TrendChart liveHealth={null} />)
    expect(screen.getByText(/No trend data yet/)).toBeDefined()
  })

  it('renders empty-state when both histories are empty', () => {
    render(<TrendChart liveHealth={{ ...base, health_history: [], memory_history: [] }} />)
    expect(screen.getByText(/No trend data yet/)).toBeDefined()
  })

  it('renders chart container with health history', () => {
    render(<TrendChart liveHealth={base} />)
    expect(screen.getAllByTestId('responsive-container').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByTestId('composed-chart').length).toBeGreaterThanOrEqual(1)
  })

  it('builds points from memory history alone when health is empty', () => {
    render(<TrendChart liveHealth={{ ...base, health_history: [] }} />)
    expect(screen.getAllByTestId('composed-chart').length).toBeGreaterThanOrEqual(1)
  })
})
