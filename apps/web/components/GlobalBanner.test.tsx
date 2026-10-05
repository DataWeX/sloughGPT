// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { GlobalBanner } from './GlobalBanner'
import { useBannerStore } from '@/lib/banner-store'

vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>
      {children}
    </button>
  ),
  StatusDot: () => <span data-testid="status-dot" />,
  IconX: () => <span data-testid="icon-x" />,
}))

afterEach(() => {
  cleanup()
  useBannerStore.getState().clearBanners()
})

describe('GlobalBanner', () => {
  it('renders nothing when no banners', () => {
    const { container } = render(<GlobalBanner />)
    expect(container.firstElementChild).toBeNull()
  })

  it('renders banner with action and dismisses on action', () => {
    const onAction = vi.fn()
    useBannerStore.getState().showBanner({
      tone: 'warning',
      title: 'Training failed',
      message: 'empty dataset',
      action: { label: 'Pick another', onAction },
    })
    render(<GlobalBanner />)
    expect(screen.getByText('Training failed')).toBeDefined()
    fireEvent.click(screen.getByText('Pick another'))
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(useBannerStore.getState().banners).toHaveLength(0)
  })

  it('dismisses via close button', () => {
    useBannerStore.getState().showBanner({ tone: 'info', title: 'Hello' })
    render(<GlobalBanner />)
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss: Hello' }))
    expect(useBannerStore.getState().banners).toHaveLength(0)
  })

  it('renders exactly one banner row when two producers raised banners', () => {
    useBannerStore.getState().showBanner({ tone: 'info', title: 'First concern' })
    useBannerStore.getState().showBanner({
      tone: 'warning',
      title: 'Backend restarting',
      message: 'reconnecting…',
    })
    render(<GlobalBanner />)
    const rows = screen.getAllByRole('alert')
    expect(rows).toHaveLength(1)
    expect(screen.getByText('Backend restarting')).toBeDefined()
    expect(screen.queryByText('First concern')).toBeNull()
  })

  it('renders the structured payload: tone, title, message, action', () => {
    const onAction = vi.fn()
    useBannerStore.getState().showBanner({
      tone: 'warning',
      title: 'Backend not responding',
      message: 'the app is running in degraded mode.',
      action: { label: 'Retry', onAction },
    })
    const { container } = render(<GlobalBanner />)
    const row = screen.getByRole('alert')
    expect(row.getAttribute('data-tone')).toBe('warning')
    expect(row.textContent).toContain(
      'Backend not responding — the app is running in degraded mode.',
    )
    fireEvent.click(screen.getByText('Retry'))
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(container.querySelector('[role="alert"]')).toBeNull()
  })
})
