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
})
