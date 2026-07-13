import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

import { TrainingProgress, TrainingCompleteBanner, TrainingErrorBanner } from './TrainingStatus'

describe('TrainingProgress', () => {
  afterEach(cleanup)

  it('renders status text', () => {
    render(<TrainingProgress status="Training epoch 3/10..." />)
    expect(screen.getByText('Training epoch 3/10...')).toBeDefined()
  })

  it('has correct aria attributes', () => {
    render(<TrainingProgress status="Working..." />)
    const el = screen.getByRole('status')
    expect(el.getAttribute('aria-live')).toBe('polite')
    expect(el.getAttribute('aria-label')).toBe('Training progress')
  })

  it('renders progress bar', () => {
    const { container } = render(<TrainingProgress status="..." />)
    const bar = container.querySelector('.animate-pulse')
    expect(bar).toBeTruthy()
  })
})

describe('TrainingCompleteBanner', () => {
  afterEach(cleanup)

  it('renders default message', () => {
    render(<TrainingCompleteBanner />)
    expect(screen.getByText('Training complete!')).toBeDefined()
  })

  it('renders custom message', () => {
    render(<TrainingCompleteBanner message="Done!" />)
    expect(screen.getByText('Done!')).toBeDefined()
  })

  it('renders explanation when provided', () => {
    render(<TrainingCompleteBanner explanation="Loss: 0.5" />)
    expect(screen.getByText('Loss: 0.5')).toBeDefined()
  })

  it('hides explanation when not provided', () => {
    const { container } = render(<TrainingCompleteBanner />)
    expect(container.querySelector('.text-xs.text-muted-foreground')).toBeNull()
  })

  it('renders Load for chat button when onLoad provided', () => {
    const onLoad = vi.fn()
    render(<TrainingCompleteBanner onLoad={onLoad} />)
    expect(screen.getByText('Load for chat')).toBeDefined()
  })

  it('hides Load for chat button when onLoad not provided', () => {
    render(<TrainingCompleteBanner />)
    expect(screen.queryByText('Load for chat')).toBeNull()
  })

  it('renders Train another button when onReset provided', () => {
    const onReset = vi.fn()
    render(<TrainingCompleteBanner onReset={onReset} />)
    expect(screen.getByText('Train another')).toBeDefined()
  })

  it('hides Train another button when onReset not provided', () => {
    render(<TrainingCompleteBanner />)
    expect(screen.queryByText('Train another')).toBeNull()
  })

  it('always renders Try in chat button', () => {
    render(<TrainingCompleteBanner />)
    expect(screen.getByText('Try in chat')).toBeDefined()
  })
})

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
