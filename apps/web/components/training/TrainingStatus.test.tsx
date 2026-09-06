import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

import { TrainingErrorBanner } from './TrainingStatus'

describe('TrainingErrorBanner', () => {
  afterEach(cleanup)

  it('renders error message', () => {
    render(<TrainingErrorBanner error="OOM killed" />)
    expect(screen.getByText('OOM killed')).toBeDefined()
  })

  it('renders training failed heading', () => {
    render(<TrainingErrorBanner error="..." />)
    expect(screen.getByText('Training failed')).toBeDefined()
  })

  it('renders Retry button when onRetry provided', () => {
    const onRetry = vi.fn()
    render(<TrainingErrorBanner error="..." onRetry={onRetry} />)
    expect(screen.getByText('Retry')).toBeDefined()
  })

  it('hides Retry button when onRetry not provided', () => {
    render(<TrainingErrorBanner error="..." />)
    expect(screen.queryByText('Retry')).toBeNull()
  })

  it('renders Dismiss button when onDismiss provided', () => {
    const onDismiss = vi.fn()
    render(<TrainingErrorBanner error="..." onDismiss={onDismiss} />)
    expect(screen.getByText('Dismiss')).toBeDefined()
  })

  it('hides Dismiss button when onDismiss not provided', () => {
    render(<TrainingErrorBanner error="..." />)
    expect(screen.queryByText('Dismiss')).toBeNull()
  })
})
