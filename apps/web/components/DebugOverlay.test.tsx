import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

const mockUseErrorStore = vi.hoisted(() => {
  const fn = vi.fn() as unknown as MockUseErrorStore
  fn.getState = vi.fn(() => ({ errors: [] }))
  return fn
})

type MockUseErrorStore = {
  (sel: (s: unknown) => unknown): unknown
  getState: () => unknown
  mockImplementation: (impl: (sel: (s: unknown) => unknown) => unknown) => void
}

vi.mock('@/lib/error-store', () => ({
  useErrorStore: mockUseErrorStore,
}))

const mockUseLiveStatus = vi.hoisted(() => vi.fn())
vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: mockUseLiveStatus,
  useLiveStatusStore: vi.fn(() => ({ connectionStatus: 'connected' })),
}))

const mockDebugData = {
  model_loaded: true,
  model_type: 'gpt2',
  soul: 'friendly',
  uptime_seconds: 3600,
  request_count: 150,
  error_count: 2,
  inference_count: 75,
  total_tokens: 45000,
  tokens_per_sec: 12.5,
  avg_tokens_per_request: 600,
  avg_latency_ms: 250,
  requests_per_minute: 5,
  health_score: {
    score: 85,
    status: 'healthy',
    error_rate_score: 90,
    latency_score: 80,
    throughput_score: 85,
    uptime_score: 90,
  },
  model_metrics: [
    { model: 'gpt2', count: 75, total_tokens: 45000, tokens_per_sec: 12.5, avg_tokens: 600 },
  ],
  model_events: [
    { type: 'load', model: 'gpt2', detail: 'loaded', ts: 1000 },
  ],
  health_history: [
    { score: 80, status: 'healthy', ts: 1000 },
    { score: 85, status: 'healthy', ts: 2000 },
  ],
  memory_history: [
    { rss_mb: 512, virtual_mb: 1024, system_percent: 25, ts: 1000 },
    { rss_mb: 520, virtual_mb: 1024, system_percent: 26, ts: 2000 },
  ],
  rate_violations: [
    { path: '/chat/stream', count: 3, limit: 10, ts: 1000 },
  ],
  path_latencies: [
    { path: '/chat/stream', avg_ms: 250, count: 75, p95_ms: 400 },
  ],
  recent_errors: [
    { path: '/chat/stream', method: 'POST', status: 500, message: 'timeout', error_type: 'TimeoutError', ts: 1000 },
  ],
  recent_requests: [
    { path: '/chat/stream', method: 'POST', status: 200, elapsed_ms: 250 },
  ],
  cpu_percent: 45,
  memory_percent: 60,
  gpu_backend: null,
}

import { DebugOverlay } from './DebugOverlay'

describe('DebugOverlay', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockFetch.mockReset()
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockDebugData),
    })
    mockUseErrorStore.mockImplementation((sel: (s: unknown) => unknown) => {
      const state = { errors: [] }
      return sel(state)
    })
    mockUseErrorStore.getState = vi.fn(() => ({ errors: [] }))
    mockUseLiveStatus.mockReturnValue({ health: mockDebugData })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('returns null when closed', () => {
    const { container } = render(<DebugOverlay open={false} onOpenChange={() => {}} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders debug panel when open', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Debug')).toBeDefined()
    })
  })

  it('shows health score section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('HEALTH')).toBeDefined()
    })
    const score85 = screen.getAllByText('85')
    expect(score85.length).toBeGreaterThanOrEqual(1)
  })

  it('shows status fields from live data', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      const gpt2s = screen.getAllByText('gpt2')
      expect(gpt2s.length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getByText('friendly')).toBeDefined()
    expect(screen.getByText('3600s')).toBeDefined()
    expect(screen.getByText('150')).toBeDefined()
  })

  it('shows model metrics section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Models')).toBeDefined()
    })
  })

  it('shows model events section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Model events')).toBeDefined()
    })
  })

  it('shows slowest endpoints section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Slowest endpoints')).toBeDefined()
    })
    const streams = screen.getAllByText('/chat/stream')
    expect(streams.length).toBeGreaterThanOrEqual(1)
  })

  it('shows memory section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText(/Memory/)).toBeDefined()
    })
  })

  it('shows rate violations section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Rate limit hits')).toBeDefined()
    })
  })

  it('shows server errors section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Server errors')).toBeDefined()
    })
    expect(screen.getByText('TimeoutError')).toBeDefined()
  })

  it('shows recent requests section', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Recent requests')).toBeDefined()
    })
  })

  it('shows last frontend error', async () => {
    const errState = {
      errors: [
        { id: '1', title: 'Network Error', message: 'Failed to fetch', severity: 'error', timestamp: 1000, requestId: 'req-123' },
      ],
    }
    mockUseErrorStore.mockImplementation((sel: (s: unknown) => unknown) => sel(errState))
    mockUseErrorStore.getState = vi.fn(() => errState)
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText(/Last FE error/)).toBeDefined()
    })
  })

  it('closes on close button click', () => {
    const onClose = vi.fn()
    render(<DebugOverlay open={true} onOpenChange={onClose} />)
    const closeBtn = screen.getByLabelText('Close debug overlay')
    fireEvent.click(closeBtn)
    expect(onClose).toHaveBeenCalledWith(false)
  })

  it('fetches debug data on mount', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalled()
    })
    const callUrl = mockFetch.mock.calls[0][0] as string
    expect(callUrl).toContain('/health/debug')
  })

  it('does not fetch when closed', () => {
    render(<DebugOverlay open={false} onOpenChange={() => {}} />)
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('shows placeholder values when fetch fails', async () => {
    mockFetch.mockRejectedValue(new Error('network error'))
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('Debug')).toBeDefined()
    })
    const dashEls = screen.getAllByText('—')
    expect(dashEls.length).toBeGreaterThan(0)
  })

  it('shows health score sub-scores', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.getByText('err')).toBeDefined()
      expect(screen.getByText('lat')).toBeDefined()
      expect(screen.getByText('tps')).toBeDefined()
      expect(screen.getByText('up')).toBeDefined()
    })
  })

  it('renders sparkline when health history has enough data', async () => {
    render(<DebugOverlay open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      const svg = document.querySelector('svg')
      expect(svg).not.toBeNull()
    })
  })
})
