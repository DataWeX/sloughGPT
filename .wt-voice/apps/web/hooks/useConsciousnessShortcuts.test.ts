import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

const mockPush = vi.fn()
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))
vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: vi.fn() }),
}))

function fireKey(key: string) {
  window.dispatchEvent(new KeyboardEvent('keydown', {
    key,
    ctrlKey: true,
    shiftKey: true,
    bubbles: true,
  }))
}

let mod: typeof import('./useConsciousnessShortcuts')

describe('useConsciousnessShortcuts', () => {
  beforeEach(async () => {
    mockPush.mockClear()
    mod = await import('./useConsciousnessShortcuts')
  })

  it('navigates to dashboard on Ctrl+Shift+D', () => {
    const { result } = renderHook(() => mod.useConsciousnessShortcuts())
    act(() => { fireKey('D') })
    expect(mockPush).toHaveBeenCalledWith('/consciousness/dashboard')
    expect(result.current.lastAction).toBe('dashboard')
  })

  it('navigates to personality on Ctrl+Shift+P', () => {
    const { result } = renderHook(() => mod.useConsciousnessShortcuts())
    act(() => { fireKey('P') })
    expect(mockPush).toHaveBeenCalledWith('/consciousness/personality')
    expect(result.current.lastAction).toBe('personality')
  })

  it('does nothing when disabled', () => {
    const { result } = renderHook(() => mod.useConsciousnessShortcuts(false))
    act(() => { fireKey('D') })
    expect(result.current.lastAction).toBeNull()
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('ignores shortcuts when focus is in input', () => {
    renderHook(() => mod.useConsciousnessShortcuts())
    const input = document.createElement('input')
    document.body.appendChild(input)
    input.focus()

    const event = new KeyboardEvent('keydown', {
      key: 'D', ctrlKey: true, shiftKey: true, bubbles: true,
    })
    Object.defineProperty(event, 'target', { value: input })
    window.dispatchEvent(event)

    expect(mockPush).not.toHaveBeenCalled()
    document.body.removeChild(input)
  })

  it('registerShortcuts toggles registration', () => {
    const { result } = renderHook(() => mod.useConsciousnessShortcuts(true))
    act(() => { result.current.registerShortcuts(false) })
    act(() => { fireKey('D') })
    expect(mockPush).not.toHaveBeenCalled()

    act(() => { result.current.registerShortcuts(true) })
    act(() => { fireKey('D') })
    expect(mockPush).toHaveBeenCalled()
  })
})
