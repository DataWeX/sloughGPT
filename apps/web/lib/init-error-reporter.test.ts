// @vitest-environment node
import { describe, expect, it, vi } from 'vitest'

const mockAddEventListener = vi.fn()

vi.stubGlobal('window', {
  addEventListener: mockAddEventListener,
  location: { href: 'http://localhost:3000' },
  localStorage: { getItem: vi.fn(() => '[]'), setItem: vi.fn(), clear: vi.fn() },
})

import { initErrorReporter } from './error-reporter'

describe('initErrorReporter', () => {
  it('registers listeners only on first call', () => {
    initErrorReporter()
    expect(mockAddEventListener).toHaveBeenCalledWith('error', expect.any(Function))
    expect(mockAddEventListener).toHaveBeenCalledWith('unhandledrejection', expect.any(Function))
    expect(mockAddEventListener).toHaveBeenCalledWith('beforeunload', expect.any(Function))
    expect(mockAddEventListener).toHaveBeenCalledTimes(4)

    mockAddEventListener.mockClear()
    initErrorReporter()
    expect(mockAddEventListener).not.toHaveBeenCalled()
  })
})
