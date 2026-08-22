/**
 * Tests for useShell hook.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useShell } from './useShell'

vi.mock('@/lib/shell-controller', () => ({
  shellExec: vi.fn(),
  shellExecStream: vi.fn(),
}))

import { shellExec, shellExecStream } from '@/lib/shell-controller'

describe('useShell', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('initializes with empty state', () => {
    const { result } = renderHook(() => useShell())
    expect(result.current.state.lines).toEqual([])
    expect(result.current.state.isRunning).toBe(false)
    expect(result.current.state.exitCode).toBeNull()
    expect(result.current.state.error).toBeNull()
  })

  it('execute runs non-streaming command', async () => {
    vi.mocked(shellExec).mockResolvedValue({
      output: 'hello\n',
      exit_code: 0,
      elapsed_ms: 1.0,
    })

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('echo hello', false)
    })

    expect(result.current.state.lines).toHaveLength(1)
    expect(result.current.state.lines[0].text).toBe('hello')
    expect(result.current.state.exitCode).toBe(0)
    expect(result.current.state.isRunning).toBe(false)
  })

  it('execute handles errors gracefully', async () => {
    vi.mocked(shellExec).mockRejectedValue(new Error('Network error'))

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('bad cmd', false)
    })

    expect(result.current.state.error).toBe('Network error')
    expect(result.current.state.isRunning).toBe(false)
  })

  it('clear resets state', async () => {
    vi.mocked(shellExec).mockResolvedValue({
      output: 'data\n',
      exit_code: 0,
      elapsed_ms: 0,
    })

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('echo data', false)
    })

    expect(result.current.state.lines).toHaveLength(1)

    act(() => {
      result.current.clear()
    })

    expect(result.current.state.lines).toEqual([])
    expect(result.current.state.exitCode).toBeNull()
  })

  it('does not execute empty commands', async () => {
    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('  ', false)
    })

    expect(shellExec).not.toHaveBeenCalled()
  })

  it('cancel stops execution', async () => {
    const { result } = renderHook(() => useShell())

    act(() => {
      result.current.cancel()
    })

    expect(result.current.state.isRunning).toBe(false)
  })

  it('maxLines caps state.lines to specified limit', async () => {
    vi.mocked(shellExec).mockResolvedValue({
      output: 'line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n',
      exit_code: 0,
      elapsed_ms: 0,
    })

    const { result } = renderHook(() => useShell(3))

    await act(async () => {
      await result.current.execute('echo multi', false)
    })

    expect(result.current.state.lines).toHaveLength(3)
    expect(result.current.state.lines[0].text).toBe('line8')
    expect(result.current.state.lines[1].text).toBe('line9')
    expect(result.current.state.lines[2].text).toBe('line10')
  })

  it('streaming execution calls shellExecStream with callbacks', async () => {
    let onLineCb: ((line: string, index: number) => void) | undefined
    let onCompleteCb: ((exitCode: number, totalLines: number, elapsedMs: number) => void) | undefined

    vi.mocked(shellExecStream).mockImplementation(async (_cmd, callbacks) => {
      onLineCb = callbacks.onLine
      onCompleteCb = callbacks.onComplete
      onLineCb?.('streamed line', 0)
      onLineCb?.('second line', 1)
      onCompleteCb?.(0, 2, 1.5)
    })

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('echo streamed', true)
    })

    expect(shellExecStream).toHaveBeenCalled()
    expect(result.current.state.lines).toHaveLength(2)
    expect(result.current.state.lines[0].text).toBe('streamed line')
    expect(result.current.state.lines[1].text).toBe('second line')
    expect(result.current.state.exitCode).toBe(0)
  })

  it('streaming execution handles errors via onError', async () => {
    vi.mocked(shellExecStream).mockImplementation(async (_cmd, callbacks) => {
      callbacks.onError?.('Stream failed')
    })

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('bad stream', true)
    })

    expect(result.current.state.error).toBe('Stream failed')
    expect(result.current.state.isRunning).toBe(false)
  })

  it('maxLines defaults to 1000', async () => {
    const lines = Array.from({ length: 100 }, (_, i) => `line${i}`).join('\n') + '\n'
    vi.mocked(shellExec).mockResolvedValue({
      output: lines,
      exit_code: 0,
      elapsed_ms: 0,
    })

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('echo many', false)
    })

    // 100 lines with default maxLines=1000, all should be kept
    expect(result.current.state.lines).toHaveLength(100)
  })

  it('aborts in-flight request on unmount', async () => {
    let signalFromExec: AbortSignal | undefined
    vi.mocked(shellExecStream).mockImplementation(async (_cmd, _callbacks, signal) => {
      signalFromExec = signal
      await new Promise(() => {}) // hang forever
    })

    const { result, unmount } = renderHook(() => useShell())

    act(() => {
      result.current.execute('long cmd', true)
    })

    // Wait for execute to start
    await new Promise(r => setTimeout(r, 10))
    expect(signalFromExec).toBeDefined()
    expect(signalFromExec!.aborted).toBe(false)

    unmount()

    // After unmount, signal should be aborted
    expect(signalFromExec!.aborted).toBe(true)
  })

  it('does not show error when sync path is aborted', async () => {
    vi.mocked(shellExec).mockImplementation(async () => {
      const err = new DOMException('The user aborted a request.', 'AbortError')
      throw err
    })

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('slow cmd', false)
    })

    expect(result.current.state.error).toBeNull()
    expect(result.current.state.isRunning).toBe(false)
  })
})
