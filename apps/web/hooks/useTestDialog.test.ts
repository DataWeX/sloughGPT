/**
 * @vitest-environment jsdom
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
    expect(result.current.testOutput).toBe('')
    expect(result.current.testLoading).toBe(false)
  })

  it('setter functions update individual states', () => {
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestDialogOpen(true))
    expect(result.current.testDialogOpen).toBe(true)
    act(() => result.current.setTestPrompt('hi'))
    expect(result.current.testPrompt).toBe('hi')
    act(() => result.current.setTestOutput('hello'))
    expect(result.current.testOutput).toBe('hello')
  })

  it('clearTest resets prompt and output', () => {
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    act(() => result.current.setTestOutput('hello'))
    act(() => result.current.clearTest())
    expect(result.current.testPrompt).toBe('')
    expect(result.current.testOutput).toBe('')
  })

  it('handleTestModel does nothing when prompt is empty', async () => {
    const { result } = renderHook(() => useTestDialog())
    await act(async () => { await result.current.handleTestModel() })
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('handleTestModel calls /inference/generate and sets output', async () => {
    mockFetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({ text: 'hello world' }) })
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    await act(async () => { await result.current.handleTestModel() })
    expect(mockFetch).toHaveBeenCalledTimes(1)
    expect(result.current.testOutput).toBe('hello world')
  })

  it('handleTestModel shows error when fetch fails', async () => {
    mockFetch.mockRejectedValue(new Error('network down'))
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    await act(async () => { await result.current.handleTestModel() })
    expect(result.current.testOutput).toBe('Error: network down')
  })

  it('handleTestModel shows error when response not ok', async () => {
    mockFetch.mockResolvedValue({ ok: false })
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    await act(async () => { await result.current.handleTestModel() })
    expect(result.current.testOutput).toBe('Error: Inference failed')
  })
})
