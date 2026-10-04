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

import { useBannerStore } from '@/lib/banner-store'
import { logStateEvent } from '@/lib/state-events'
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
  useBannerStore.getState().clearBanners()
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
    const { container } = render(<StartupOverlay />)
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByRole('heading')).toHaveTextContent('Connecting')

    act(() => {
      vi.advanceTimersByTime(8_000)
    })
    expect(screen.getByRole('alert')).toBeTruthy()
    expect(screen.getByRole('heading')).toHaveTextContent('Still connecting')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    // De-stacked stuck variant: a "no response" screen must not also claim
    // progress, so the shimmer bar and the stage dots stay out of it.
    expect(screen.queryByRole('progressbar', { name: 'Startup progress' })).toBeNull()
    expect(container.querySelector('[role="group"]')).toBeNull()
    expect(screen.getByRole('status').textContent).toContain('no response')

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

  it('force-dismisses a stalled known stage and raises the degraded banner', () => {
    useLiveStatusMock.mockReturnValue(status({ startupStage: 'init', connected: true }))
    render(<StartupOverlay />)
    expect(screen.getByRole('progressbar', { name: 'Startup progress' })).toBeTruthy()

    act(() => {
      vi.advanceTimersByTime(20_000)
    })
    // Banner raised the moment we give up; the overlay is mid-fade.
    const banners = useBannerStore.getState().banners
    expect(banners).toHaveLength(1)
    expect(banners[0].key).toBe('backend-connection')
    expect(banners[0].tone).toBe('warning')
    expect(banners[0].action?.label).toBe('Retry')
    expect(logStateEvent).toHaveBeenCalledWith('overlay_timeout', expect.anything())
    expect(screen.getByRole('progressbar', { name: 'Startup progress' })).toBeTruthy()

    act(() => {
      vi.advanceTimersByTime(600)
    })
    expect(screen.queryByRole('progressbar', { name: 'Startup progress' })).toBeNull()
    // No stuck UI for a known stage — the stall bound is the exit that fired.
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('keeps the overlay up while the boot is still progressing', () => {
    useLiveStatusMock.mockReturnValue(status({ startupStage: 'init', connected: true }))
    const { rerender } = render(<StartupOverlay />)

    act(() => {
      vi.advanceTimersByTime(15_000)
    })
    expect(screen.getByRole('progressbar', { name: 'Startup progress' })).toBeTruthy()

    useLiveStatusMock.mockReturnValue(status({ startupStage: 'critical', connected: true }))
    rerender(<StartupOverlay />)
    act(() => {
      vi.advanceTimersByTime(15_000)
    })
    // 30s elapsed > the 20s bound, but the stage change restarted the clock.
    expect(screen.getByRole('progressbar', { name: 'Startup progress' })).toBeTruthy()
    expect(useBannerStore.getState().banners).toHaveLength(0)

    useLiveStatusMock.mockReturnValue(
      status({ startupStage: 'critical', startupModelProgress: 0.5, connected: true }),
    )
    rerender(<StartupOverlay />)
    act(() => {
      vi.advanceTimersByTime(15_000)
    })
    expect(screen.getByRole('progressbar', { name: 'Startup progress' })).toBeTruthy()
    expect(useBannerStore.getState().banners).toHaveLength(0)

    // Ready arrives → the happy path dismisses with no banner.
    useLiveStatusMock.mockReturnValue(status({ startupStage: 'ready', connected: true }))
    rerender(<StartupOverlay />)
    act(() => {
      vi.advanceTimersByTime(600)
    })
    expect(screen.queryByRole('progressbar', { name: 'Startup progress' })).toBeNull()
    expect(useBannerStore.getState().banners).toHaveLength(0)
    expect(logStateEvent).toHaveBeenCalledWith('overlay_hidden', expect.anything())
    expect(logStateEvent).not.toHaveBeenCalledWith('overlay_timeout', expect.anything())
  })

  it('gives up at the stall bound when the stage never becomes known', () => {
    useLiveStatusMock.mockReturnValue(status())
    render(<StartupOverlay />)

    act(() => {
      vi.advanceTimersByTime(8_000)
    })
    // The existing stuck UI still fires first.
    expect(screen.getByRole('alert')).toBeTruthy()

    act(() => {
      vi.advanceTimersByTime(12_000)
    })
    const banners = useBannerStore.getState().banners
    expect(banners).toHaveLength(1)
    expect(banners[0].key).toBe('backend-connection')

    act(() => {
      vi.advanceTimersByTime(600)
    })
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.queryByRole('progressbar', { name: 'Startup progress' })).toBeNull()
  })
})
