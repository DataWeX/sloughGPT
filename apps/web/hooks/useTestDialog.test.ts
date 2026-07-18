/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useTestDialog } from './useTestDialog'

const mockGenerate = vi.fn()
vi.mock('@/lib/generate-controller', () => ({
  generateController: { generate: (...args: unknown[]) => mockGenerate(...args) },
}))

afterEach(() => {
  cleanup()
  mockGenerate.mockReset()
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
    expect(mockGenerate).not.toHaveBeenCalled()
  })

  it('handleTestModel calls generate and sets structured result', async () => {
    mockGenerate.mockResolvedValue({ text: 'hello world', model: 'gpt2', tokens_generated: 3 })
    const { result } = renderHook(() => useTestDialog())
    act(() => result.current.setTestPrompt('hi'))
    await act(async () => { await result.current.handleTestModel() })
    expect(mockGenerate).toHaveBeenCalledTimes(1)
    expect(result.current.testResult).toEqual({
      prompt: 'hi',
      response: 'hello world',
      model: 'gpt2',
      tokens_generated: 3,
      error: '',
    })
  })

  it('handleTestModel shows error when generate rejects', async () => {
    mockGenerate.mockRejectedValue(new Error('network down'))
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

  it('handleTestModel shows error when generate throws error field', async () => {
    mockGenerate.mockRejectedValue(new Error('model not loaded'))
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
