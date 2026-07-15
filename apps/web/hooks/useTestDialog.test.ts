/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useTestDialog } from './useTestDialog'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

afterEach(() => {
  cleanup()
  mockFetch.mockReset()
})

describe('useTestDialog', () => {
  it('returns default state', () => {
    const { result } = renderHook(() => useTestDialog())
    expect(result.current.testDialogOpen).toBe(false)
    expect(result.current.testPrompt).toBe('')
    expect(result.current.testResult).toBeNull()
    expect(result.current.testLoading).toBe(false)
  })

  it('setter functions update individual states', () => {
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestDialogOpen(true))
    expect(result.current.testDialogOpen).toBe(true)
    act(() => result.current.setTestPrompt('hi'))
    expect(result.current.testPrompt).toBe('hi')
  })

  it('clearTest resets prompt and result', () => {
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    act(() => result.current.clearTest())
    expect(result.current.testPrompt).toBe('')
    expect(result.current.testResult).toBeNull()
  })

  it('handleTestModel does nothing when prompt is empty', async () => {
    const { result } = renderHook(() => useTestDialog())
    await act(async () => { await result.current.handleTestModel() })
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('handleTestModel calls /inference/generate and sets structured result', async () => {
    mockFetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({ text: 'hello world', model: 'gpt2', tokens_generated: 3 }) })
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    await act(async () => { await result.current.handleTestModel() })
    expect(mockFetch).toHaveBeenCalledTimes(1)
    expect(result.current.testResult).toEqual({
      prompt: 'hi',
      response: 'hello world',
      model: 'gpt2',
      tokens_generated: 3,
      error: '',
    })
  })

  it('handleTestModel shows error when fetch fails', async () => {
    mockFetch.mockRejectedValue(new Error('network down'))
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    await act(async () => { await result.current.handleTestModel() })
    expect(result.current.testResult).toEqual({
      prompt: 'hi',
      response: '',
      model: '',
      tokens_generated: 0,
      error: 'network down',
    })
  })

  it('handleTestModel shows error when response not ok', async () => {
    mockFetch.mockResolvedValue({ ok: false, json: () => Promise.resolve({ detail: 'model not loaded' }) })
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    await act(async () => { await result.current.handleTestModel() })
    expect(result.current.testResult).toEqual({
      prompt: 'hi',
      response: '',
      model: '',
      tokens_generated: 0,
      error: 'model not loaded',
    })
  })
})
