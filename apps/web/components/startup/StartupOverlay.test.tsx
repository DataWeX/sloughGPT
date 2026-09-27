// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react'
import React from 'react'

const useLiveStatusMock = vi.fn()

vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => useLiveStatusMock(),
}))

vi.mock('@/lib/state-events', () => ({
  logStateEvent: vi.fn(),
}))

import { StartupOverlay } from './StartupOverlay'

function status(overrides: Record<string, unknown> = {}) {
  return {
    startupStage: 'unknown',
    startupModelProgress: 0,
    startupModelProgressMessage: '',
    startupHooks: {},
    startupElapsed: 0,
    connected: false,
    ...overrides,
  }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('StartupOverlay', () => {
  it('shows the Man brand mark', () => {
    useLiveStatusMock.mockReturnValue(status())
    render(<StartupOverlay />)
    expect(screen.getByTestId('man-mark')).toHaveTextContent('M')
  })

  it('shows indeterminate shimmer bar when stage is unknown', () => {
    useLiveStatusMock.mockReturnValue(status())
    const { container } = render(<StartupOverlay />)
    expect(container.querySelector('.sl-bar-shimmer')).toBeTruthy()
    const bar = screen.getByRole('progressbar', { name: 'Startup progress' })
    expect(bar).not.toHaveAttribute('aria-valuenow')
  })

  it('shows determinate bar with aria-valuenow when stage is known', () => {
    useLiveStatusMock.mockReturnValue(status({ startupStage: 'init' }))
    const { container } = render(<StartupOverlay />)
    expect(container.querySelector('.sl-bar-shimmer')).toBeNull()
    expect(screen.getByRole('progressbar', { name: 'Startup progress' })).toHaveAttribute(
      'aria-valuenow',
      '25',
    )
  })

  it('animates stage dots as a wave when connecting', () => {
    useLiveStatusMock.mockReturnValue(status())
    const { container } = render(<StartupOverlay />)
    const dots = container.querySelectorAll('[role="img"]')
    expect(dots).toHaveLength(4)
    dots.forEach((dot) => {
      expect(dot.className).toContain('sl-dot-wave')
      expect(dot.getAttribute('aria-label')).toMatch(/Stage \d of 4:/)
    })
    expect(screen.getByRole('status').textContent).toBe('Connecting')
  })

  it('marks completed and active stage dots', () => {
    useLiveStatusMock.mockReturnValue(status({ startupStage: 'critical' }))
    const { container } = render(<StartupOverlay />)
    const dots = Array.from(container.querySelectorAll('[role="img"]'))
    expect(dots[0].className).toContain('bg-[#28c840]')
    expect(dots[1].className).toContain('sl-dot-pulse')
    expect(dots[2].className).toContain('bg-[#2c2c2e]')
    expect(screen.getByRole('status').textContent).toBe('Stage 2 of 4: Starting core services')
  })

  it('shows stuck-connecting recovery after timeout and reloads on Retry', () => {
    useLiveStatusMock.mockReturnValue(status())
    const reload = vi.fn()
    Object.defineProperty(window, 'location', {
      value: { ...window.location, reload },
      writable: true,
    })
    render(<StartupOverlay />)
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByRole('heading')).toHaveTextContent('Connecting')

    act(() => {
      vi.advanceTimersByTime(8_000)
    })
    expect(screen.getByRole('alert')).toBeTruthy()
    expect(screen.getByRole('heading')).toHaveTextContent('Still connecting')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('does not show stuck UI when stage is known', () => {
    useLiveStatusMock.mockReturnValue(status({ startupStage: 'init' }))
    render(<StartupOverlay />)
    act(() => {
      vi.advanceTimersByTime(8_000)
    })
    expect(screen.queryByRole('alert')).toBeNull()
  })
})
