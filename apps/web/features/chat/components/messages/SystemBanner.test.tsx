/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { SystemBanner } from './SystemBanner'

afterEach(cleanup)

describe('SystemBanner', () => {
  it('renders title', () => {
    render(<SystemBanner type="info" title="System update" />)
    expect(screen.getByText('System update')).toBeTruthy()
  })

  it('renders message when provided', () => {
    render(<SystemBanner type="warning" title="Warning" message="Something happened" />)
    expect(screen.getByText('Something happened')).toBeTruthy()
  })

  it('renders action button when actionLabel and onAction provided', () => {
    render(<SystemBanner type="offline" title="Offline" actionLabel="Retry" onAction={() => {}} />)
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('calls onAction when action button is clicked', () => {
    const onAction = vi.fn()
    render(<SystemBanner type="offline" title="Offline" actionLabel="Retry" onAction={onAction} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onAction).toHaveBeenCalledTimes(1)
  })

  it('has alert role with aria-live assertive', () => {
    const { container } = render(<SystemBanner type="info" title="Info" />)
    const el = container.firstElementChild!
    expect(el.getAttribute('role')).toBe('alert')
    expect(el.getAttribute('aria-live')).toBe('assertive')
  })

  it('renders variant-specific icons', () => {
    const { container: c1 } = render(<SystemBanner type="info" title="Info" />)
    const { container: c2 } = render(<SystemBanner type="warning" title="Warning" />)
    const { container: c3 } = render(<SystemBanner type="offline" title="Offline" />)
    const svgCount = (c: HTMLElement) => c.querySelectorAll('svg').length
    expect(svgCount(c1)).toBe(1)
    expect(svgCount(c2)).toBe(1)
    expect(svgCount(c3)).toBe(1)
  })
})
