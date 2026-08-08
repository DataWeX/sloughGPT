import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { liveStatusStore, useLiveStatus, mapDetailedToSnapshot, initLiveStatus } from './useLiveStatus'

const mocks = vi.hoisted(() => ({
  createSSEStream: vi.fn(),
  getDetailedHealth: vi.fn().mockResolvedValue(null),
}))

vi.mock('@/lib/sse-client', () => ({
  createSSEStream: mocks.createSSEStream,
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    getHealth: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    getDetailedHealth: mocks.getDetailedHealth,
  },
}))

describe('liveStatusStore', () => {
  beforeEach(() => {
    liveStatusStore.setState({
      connectionStatus: 'connecting',
      health: null,
      healthLegacy: null,
      lastUpdate: null,
      failureCount: 0,
      lastError: null,
    })
  })

  it('has correct initial state', () => {
    const state = liveStatusStore.getState()
    expect(state.connectionStatus).toBe('connecting')
    expect(state.health).toBeNull()
    expect(state.healthLegacy).toBeNull()
    expect(state.failureCount).toBe(0)
    expect(state.lastError).toBeNull()
  })

  it('setHealth updates health, lastUpdate, resets failureCount', () => {
    const snap = {
      model_loaded: true,
      model_loading: false,
      model_type: 'gpt2',
      soul: 'friendly',
      is_inferencing: false,
      inference_count: 5,
      uptime_seconds: 120,
      request_count: 10,
      error_count: 0,
      tokens_per_sec: 12.5,
      avg_latency_ms: 80,
      cpu_percent: 45,
      memory_percent: 60,
      health_score: 85,
      health_status: 'healthy',
      health_summary: 'All good',
      diagnoses: [],
      num_parameters: 124000000,
      quantization: null,
      training_pool: null,
    } as any
    act(() => { liveStatusStore.getState().setHealth(snap) })
    const state = liveStatusStore.getState()
    expect(state.health).toEqual(snap)
    expect(state.lastUpdate).toBeTypeOf('number')
    expect(state.failureCount).toBe(0)
  })

  it('setConnectionStatus updates status', () => {
    act(() => { liveStatusStore.getState().setConnectionStatus('connected') })
    expect(liveStatusStore.getState().connectionStatus).toBe('connected')
  })

  it('incrementFailures increments count', () => {
    act(() => { liveStatusStore.getState().incrementFailures() })
    act(() => { liveStatusStore.getState().incrementFailures() })
    expect(liveStatusStore.getState().failureCount).toBe(2)
  })

  it('reset restores initial state', () => {
    act(() => {
      liveStatusStore.getState().setConnectionStatus('connected')
      liveStatusStore.getState().incrementFailures()
    })
    act(() => { liveStatusStore.getState().reset() })
    const state = liveStatusStore.getState()
    expect(state.connectionStatus).toBe('connecting')
    expect(state.health).toBeNull()
    expect(state.failureCount).toBe(0)
  })

  it('setHealthLegacy updates legacy shape', () => {
    const legacy = { status: 'ok', model_loaded: true, model_type: 'gpt2', summary: 'good', inference_count: 1, is_inferencing: false } as any
    act(() => { liveStatusStore.getState().setHealthLegacy(legacy) })
    expect(liveStatusStore.getState().healthLegacy).toEqual(legacy)
  })
})

describe('mapDetailedToSnapshot', () => {
  it('maps a full detailed health response onto the snapshot shape', () => {
    const d = {
      status: 'healthy',
      uptime_seconds: 500,
      timestamp: '2026-08-06T00:00:00',
      request_count: 10,
      error_count: 2,
      avg_latency_ms: 80,
      requests_per_minute: 4.5,
      path_latencies: [{ path: '/inference/generate', avg_ms: 80, count: 5, p95_ms: 120 }],
      recent_errors: [{ path: '/chat', method: 'POST', status: 500, message: 'boom', error_type: 'Err', ts: 100 }],
      inference_count: 8,
      total_tokens: 12000,
      tokens_per_sec: 12.5,
      avg_tokens_per_request: 160,
      health_score: { score: 90, status: 'healthy' },
      status_message: 'All good',
      model_metrics: [{ model: 'gpt2', count: 3, total_tokens: 1000, tokens_per_sec: 5, avg_tokens: 333 }],
      model_events: [{ type: 'load', model: 'gpt2', detail: '', ts: 100 }],
      health_history: [{ score: 90, status: 'healthy', ts: 100 }],
      memory_history: [{ rss_mb: 300, virtual_mb: 400, system_percent: 55, ts: 100 }],
      rate_violations: [{ path: '/chat', count: 12, limit: 5, ts: 100 }],
      system: { cpu_percent: 45, memory_percent: 60, memory_available_mb: 8000 },
      model_loaded: true,
      model_type: 'gpt2',
      device: 'cpu',
      soul: 'friendly',
      inference: { is_inferencing: true, inference_count: 8 },
      quantization: null,
      training_pool: null,
    } as any

    const snap = mapDetailedToSnapshot(d)
    expect(snap.model_loaded).toBe(true)
    expect(snap.soul).toBe('friendly')
    expect(snap.device).toBe('cpu')
    expect(snap.is_inferencing).toBe(true)
    expect(snap.requests_per_minute).toBe(4.5)
    expect(snap.total_tokens).toBe(12000)
    expect(snap.cpu_percent).toBe(45)
    expect(snap.memory_percent).toBe(60)
    expect(snap.health_score).toBe(90)
    expect(snap.health_status).toBe('healthy')
    expect(snap.health_summary).toBe('All good')
    expect(snap.path_latencies).toHaveLength(1)
    expect(snap.recent_errors).toHaveLength(1)
    expect(snap.model_metrics).toHaveLength(1)
    expect(snap.model_events).toHaveLength(1)
    expect(snap.health_history).toHaveLength(1)
    expect(snap.memory_history).toHaveLength(1)
    expect(snap.rate_violations).toHaveLength(1)
  })

  it('defaults missing fields safely', () => {
    const snap = mapDetailedToSnapshot({} as any)
    expect(snap.model_loaded).toBe(false)
    expect(snap.health_score).toBe(0)
    expect(snap.health_status).toBe('unknown')
    expect(snap.diagnoses).toEqual([])
    expect(snap.path_latencies).toEqual([])
    expect(snap.recent_errors).toEqual([])
    expect(snap.cpu_percent).toBeNull()
    expect(snap.device).toBeNull()
  })
})

describe('useLiveStatus hook', () => {
  it('returns initial connecting state', () => {
    const { result } = renderHook(() => useLiveStatus())
    expect(result.current.connectionStatus).toBe('connecting')
    expect(result.current.health).toBeNull()
    expect(result.current.connected).toBe(false)
    expect(result.current.live).toBe(false)
  })

  it('returns connected when status is connected with health', () => {
    act(() => {
      liveStatusStore.getState().setConnectionStatus('connected')
      liveStatusStore.getState().setHealth({
        model_loaded: true, model_loading: false, model_type: 'gpt2', soul: null,
        is_inferencing: false, inference_count: 0, uptime_seconds: 0, request_count: 0,
        error_count: 0, tokens_per_sec: 0, avg_latency_ms: 0, cpu_percent: null,
        memory_percent: null, health_score: 80, health_status: 'ok', health_summary: '',
        diagnoses: [], num_parameters: null, quantization: null, training_pool: null,
      } as any)
    })
    const { result } = renderHook(() => useLiveStatus())
    expect(result.current.connected).toBe(true)
    expect(result.current.live).toBe(true)
    expect(result.current.health?.model_loaded).toBe(true)
  })

  it('returns offline status', () => {
    act(() => { liveStatusStore.getState().setHealthLegacy('offline') })
    const { result } = renderHook(() => useLiveStatus())
    expect(result.current.healthLegacy).toBe('offline')
  })
})

describe('initLiveStatus', () => {
  let streamConfig: any
  let streamStop: ReturnType<typeof vi.fn>

  beforeEach(() => {
    streamConfig = null
    streamStop = vi.fn()
    mocks.createSSEStream.mockImplementation((cfg: any) => {
      streamConfig = cfg
      return { start: vi.fn(), stop: streamStop }
    })
    mocks.getDetailedHealth.mockResolvedValue(null)
    liveStatusStore.setState({
      connectionStatus: 'connecting',
      health: null,
      healthLegacy: null,
      lastUpdate: null,
      failureCount: 0,
      lastError: null,
    })
  })

  it('subscribes to the health stream with reconnect', () => {
    const cleanup = initLiveStatus()
    expect(mocks.createSSEStream).toHaveBeenCalledTimes(1)
    expect(streamConfig.url).toBe('/health/stream')
    expect(streamConfig.reconnect).toBe(true)
    cleanup()
    expect(streamStop).toHaveBeenCalled()
  })

  it('updates the store from SSE health events', () => {
    const cleanup = initLiveStatus()
    streamConfig.onEvent({ stream: 'health', data: { model_loaded: true, model_type: 'qwen', soul: 'friendly', health_status: 'healthy' } })
    const s = liveStatusStore.getState()
    expect(s.connectionStatus).toBe('connected')
    expect(s.health?.model_type).toBe('qwen')
    expect(s.health?.soul).toBe('friendly')
    cleanup()
  })

  it('ignores non-health SSE envelopes', () => {
    const cleanup = initLiveStatus()
    streamConfig.onEvent({ stream: 'chat', data: { model_type: 'other' } })
    expect(liveStatusStore.getState().health).toBeNull()
    cleanup()
  })

  it('marks connection connected on stream open', () => {
    const cleanup = initLiveStatus()
    streamConfig.onOpen()
    expect(liveStatusStore.getState().connectionStatus).toBe('connected')
    cleanup()
  })

  it('falls back to detailed-health polling when SSE closes', async () => {
    mocks.getDetailedHealth.mockResolvedValue({
      model_loaded: true,
      model_type: 'gpt2',
      soul: 'friendly',
      health_score: { score: 90, status: 'healthy' },
      num_parameters: 124000000,
      inference: { inference_count: 3, is_inferencing: false },
      model_metrics: [],
      path_latencies: [],
      recent_errors: [],
      model_events: [],
      rate_violations: [],
      health_history: [],
      memory_history: [],
    } as any)
    const cleanup = initLiveStatus()
    streamConfig.onClose()
    await new Promise((r) => setTimeout(r, 0))
    const s = liveStatusStore.getState()
    expect(mocks.getDetailedHealth).toHaveBeenCalled()
    expect(s.connectionStatus).toBe('connected')
    expect(s.health?.model_type).toBe('gpt2')
    expect(s.health?.num_parameters).toBe(124000000)
    expect(s.health?.health_score).toBe(90)
    cleanup()
  })

  it('increments failures when the fallback poll returns nothing', async () => {
    mocks.getDetailedHealth.mockResolvedValue(null)
    const cleanup = initLiveStatus()
    streamConfig.onClose()
    await new Promise((r) => setTimeout(r, 0))
    expect(liveStatusStore.getState().failureCount).toBeGreaterThanOrEqual(1)
    expect(liveStatusStore.getState().connectionStatus).toBe('connecting')
    cleanup()
  })

  it('does not poll detailed health immediately when SSE delivers within the grace window', () => {
    vi.useFakeTimers()
    try {
      const cleanup = initLiveStatus()
      expect(mocks.getDetailedHealth).not.toHaveBeenCalled()
      streamConfig.onEvent({ stream: 'health', data: { model_type: 'qwen', health_status: 'healthy' } })
      vi.advanceTimersByTime(20000)
      expect(mocks.getDetailedHealth).not.toHaveBeenCalled()
      cleanup()
    } finally {
      vi.useRealTimers()
    }
  })

  it('starts the fallback poll after the grace window when no SSE event arrives', () => {
    vi.useFakeTimers()
    try {
      const cleanup = initLiveStatus()
      expect(mocks.getDetailedHealth).not.toHaveBeenCalled()
      vi.advanceTimersByTime(8000)
      expect(mocks.getDetailedHealth).toHaveBeenCalled()
      cleanup()
    } finally {
      vi.useRealTimers()
    }
  })
})
