import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
}))

import { StatusBanner } from './StatusBanner'

afterEach(cleanup)

describe('StatusBanner', () => {
  it('renders the message', () => {
    render(<StatusBanner message="All systems operational" />)
    expect(screen.getByText('All systems operational')).toBeInTheDocument()
  })

  it('renders with role="status" for non-error variants', () => {
    render(<StatusBanner variant="info" message="Hello" />)
    const el = screen.getByRole('status')
    expect(el).toBeInTheDocument()
  })

  it('renders with role="alert" for error variant', () => {
    render(<StatusBanner variant="error" message="Failure" />)
    const el = screen.getByRole('alert')
    expect(el).toBeInTheDocument()
    expect(screen.getByText('Failure')).toBeInTheDocument()
  })

  it('dismisses the banner when Dismiss is clicked', () => {
    const onDismiss = vi.fn()
    render(<StatusBanner message="Click to dismiss" onDismiss={onDismiss} />)
    expect(screen.getByText('Click to dismiss')).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Dismiss'))
    expect(onDismiss).toHaveBeenCalledTimes(1)
    expect(screen.queryByText('Click to dismiss')).not.toBeInTheDocument()
  })

  it('does not render Dismiss button when dismissible=false and no onDismiss', () => {
    render(<StatusBanner message="Persistent" dismissible={false} />)
    expect(screen.queryByLabelText('Dismiss')).not.toBeInTheDocument()
  })

  it('calls onRetry when Retry is clicked', () => {
    const onRetry = vi.fn()
    render(<StatusBanner message="Error" onRetry={onRetry} />)
    fireEvent.click(screen.getByLabelText('Retry'))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})
