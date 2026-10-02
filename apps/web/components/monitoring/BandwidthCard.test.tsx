import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { BandwidthCard } from './BandwidthCard'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import type { BandwidthStats } from '@/lib/system-controller'

const base: LiveHealthSnapshot = {
  model_loaded: true,
  model_loading: false,
  model_type: 'qwen',
  device: 'cpu',
  soul: null,
  startup_stage: 'ready',
  startup_stage_value: 3,
  startup_elapsed: 1,
  startup_model_progress: 1,
  startup_model_progress_message: '',
  startup_hooks: {},
  is_inferencing: false,
  inference_count: 0,
  uptime_seconds: 100,
  request_count: 5,
  error_count: 0,
  tokens_per_sec: 0,
  avg_latency_ms: 40,
  p95_latency_ms: 60,
  requests_per_minute: 1.5,
  total_tokens: 500,
  avg_tokens_per_request: 120,
  cpu_percent: 20,
  memory_percent: 40,
  health_score: 92,
  health_status: 'healthy',
  health_summary: 'ok',
  diagnoses: [],
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

const saved: BandwidthStats = {
  identity_bytes: 50 * 1024 * 1024, // 50 MB served uncompressed
  wire_bytes: 5 * 1024 * 1024, // 5 MB on the wire
  saved_bytes: 45 * 1024 * 1024,
  saved_pct: 90.0,
  compressed_responses: 8,
  identity_responses: 2,
  zstd_responses: 6,
  gzip_responses: 2,
}

describe('BandwidthCard', () => {
  afterEach(cleanup)

  it('renders nothing without live health', () => {
    const { container } = render(<BandwidthCard liveHealth={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when no gateway counters were ever mirrored', () => {
    const { container } = render(<BandwidthCard liveHealth={base} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when bandwidth is explicitly null', () => {
    const { container } = render(<BandwidthCard liveHealth={{ ...base, bandwidth: null }} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows identity, wire, and savings from the counters', () => {
    render(<BandwidthCard liveHealth={{ ...base, bandwidth: saved }} />)
    expect(screen.getByText('Edge bandwidth')).toBeInTheDocument()
    expect(screen.getByText('50.0 MB')).toBeInTheDocument()
    expect(screen.getByText('5.0 MB')).toBeInTheDocument()
    expect(screen.getByText('90.0%')).toBeInTheDocument()
    expect(screen.getByText('8 compressed / 10 responses · 6 zstd · 2 gzip')).toBeInTheDocument()
  })

  it('renders zero-traffic counters as absent rather than misleading', () => {
    const { container } = render(
      <BandwidthCard
        liveHealth={{
          ...base,
          bandwidth: { ...saved, identity_bytes: 0, wire_bytes: 0 },
        }}
      />,
    )
    expect(container.firstChild).toBeNull()
  })

  it('reports compression expansion honestly (negative savings)', () => {
    render(
      <BandwidthCard
        liveHealth={{
          ...base,
          bandwidth: {
            ...saved,
            identity_bytes: 1024,
            wire_bytes: 1034,
            saved_bytes: -10,
            saved_pct: -1.0,
          },
        }}
      />,
    )
    expect(screen.getByText('-1.0%')).toBeInTheDocument()
  })
})
