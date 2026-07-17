import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { liveStatusStore, useLiveStatus } from './useLiveStatus'

vi.mock('@/lib/sse-client', () => ({
  createSSEStream: vi.fn(() => ({
    start: vi.fn(),
    stop: vi.fn(),
  })),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    getHealth: vi.fn().mockResolvedValue(null),
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
