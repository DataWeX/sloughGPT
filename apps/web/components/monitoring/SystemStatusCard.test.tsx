import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { SystemStatusCard } from './SystemStatusCard'

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

describe('SystemStatusCard', () => {
  afterEach(cleanup)

  const renderCard = (liveHealth: any = base, detailed: any = null, connectionStatus = 'connected') =>
    render(<SystemStatusCard liveHealth={liveHealth} detailed={detailed} connectionStatus={connectionStatus as any} inferenceRate={4.5} loaded={true} />)

  it('renders healthy API and loaded model', () => {
    renderCard()
    expect(screen.getByText('Healthy')).toBeDefined()
    expect(screen.getByText('gpt2')).toBeDefined()
    expect(screen.getByText('live')).toBeDefined()
  })

  it('shows model params when available', () => {
    renderCard({ ...base, num_parameters: 124000000 })
    expect(screen.getByText('gpt2 · 124M')).toBeDefined()
  })

  it('formats parameters above 1B', () => {
    renderCard({ ...base, num_parameters: 1500000000 })
    expect(screen.getByText('gpt2 · 1.5B')).toBeDefined()
  })

  it('shows inferencing pulse when generating', () => {
    renderCard({ ...base, is_inferencing: true })
    expect(screen.getByText('inferencing')).toBeDefined()
  })

  it('does not show inferencing when idle', () => {
    renderCard()
    expect(screen.queryByText('inferencing')).toBeNull()
  })

  it('shows offline status', () => {
    renderCard(base, null, 'offline')
    expect(screen.getByText('offline')).toBeDefined()
  })

  it('falls back to detailed health when snapshot is absent', () => {
    renderCard(null, { status: 'healthy', model_loaded: true, model_type: 'qwen', num_parameters: 500000000, inference: { inference_count: 3 } })
    expect(screen.getByText('qwen · 500M')).toBeDefined()
  })

  it('shows unloaded model without params', () => {
    renderCard({ ...base, model_loaded: false })
    expect(screen.getByText('Not loaded')).toBeDefined()
  })

  it('shows loading state while a model loads', () => {
    renderCard({ ...base, model_loaded: false, model_loading: true, num_parameters: 500000000 })
    expect(screen.getByText('Loading…')).toBeDefined()
    expect(screen.queryByText('Not loaded')).toBeNull()
  })

  it('shows soul name when set', () => {
    renderCard({ ...base, soul: 'friendly' })
    expect(screen.getByText('friendly')).toBeDefined()
  })

  it('shows placeholder when no soul is set', () => {
    renderCard()
    expect(screen.getByText('—')).toBeDefined()
  })
})
