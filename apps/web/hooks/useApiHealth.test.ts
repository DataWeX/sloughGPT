/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { renderHook, act, cleanup, waitFor } from '@testing-library/react'

const mockGetHealth = vi.fn()

vi.mock('@/lib/model-controller', () => ({
  modelController: { getHealth: (...args: unknown[]) => mockGetHealth(...args) },
}))

import { inferenceHealthLabel, useApiHealth } from './useApiHealth'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('inferenceHealthLabel', () => {
  it('returns checking for null', () => {
    expect(inferenceHealthLabel(null)).toBe('checking...')
  })

  it('returns disconnected for offline', () => {
    expect(inferenceHealthLabel('offline')).toBe('disconnected')
  })

  it('returns inference ready when model_loaded', () => {
    expect(
      inferenceHealthLabel({ status: 'ok', model_loaded: true, model_type: 'gpt2', summary: 'ready' })
    ).toBe('inference ready · gpt2')
  })

  it('returns no weights when model not loaded', () => {
    expect(
      inferenceHealthLabel({ status: 'ok', model_loaded: false, model_type: 'gpt2', summary: 'ok' })
    ).toBe('connected · no weights (gpt2)')
  })
})

describe('useApiHealth', () => {
  it('re-fetches on visibilitychange to visible', async () => {
    mockGetHealth.mockResolvedValue({ status: 'ok', model_loaded: true, model_type: 'gpt2', summary: 'ready' })
    const { result } = renderHook(() => useApiHealth())
    await act(async () => { await result.current.refresh() })

    const callCount = mockGetHealth.mock.calls.length
    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true })
    document.dispatchEvent(new Event('visibilitychange'))

    await waitFor(() => expect(mockGetHealth.mock.calls.length).toBe(callCount + 1))
  })

  it('does not re-fetch on visibilitychange to hidden', async () => {
    mockGetHealth.mockResolvedValue({ status: 'ok', model_loaded: true, model_type: 'gpt2', summary: 'ready' })
    const { result } = renderHook(() => useApiHealth())
    await act(async () => { await result.current.refresh() })

    const callCount = mockGetHealth.mock.calls.length
    Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true })
    document.dispatchEvent(new Event('visibilitychange'))

    expect(mockGetHealth.mock.calls.length).toBe(callCount)
  })

  it('returns state and refresh', async () => {
    mockGetHealth.mockResolvedValue({ status: 'ok', model_loaded: true, model_type: 'gpt2', summary: 'ready' })
    const { result } = renderHook(() => useApiHealth())
    expect(result.current).toHaveProperty('state')
    expect(result.current).toHaveProperty('refresh')
    expect(typeof result.current.refresh).toBe('function')
  })

  it('refresh calls getHealth and updates state', async () => {
    mockGetHealth.mockResolvedValue({ status: 'ok', model_loaded: false, model_type: 'tinyllama', summary: 'ok' })
    const { result } = renderHook(() => useApiHealth())
    await act(async () => { await result.current.refresh() })
    expect(result.current.state).toEqual({ status: 'ok', model_loaded: false, model_type: 'tinyllama', summary: 'ok' })
  })

  it('refresh sets offline when getHealth returns null', async () => {
    mockGetHealth.mockResolvedValue(null)
    const { result } = renderHook(() => useApiHealth())
    await act(async () => { await result.current.refresh() })
    expect(result.current.state).toBe('offline')
  })
})
