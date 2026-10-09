/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ErrorBanner, getErrorInfo } from './ErrorBanner'

afterEach(cleanup)

describe('getErrorInfo', () => {
  it('maps status 0 to network error', () => {
    const r = getErrorInfo(0)
    expect(r.type).toBe('network')
    expect(r.canRetry).toBe(true)
  })

  it('maps status 404 to server error', () => {
    const r = getErrorInfo(404)
    expect(r.type).toBe('server')
  })

  it('maps status 500 to server error', () => {
    const r = getErrorInfo(500)
    expect(r.type).toBe('server')
    expect(r.canRetry).toBe(true)
  })

  it('maps status 503 to model error', () => {
    const r = getErrorInfo(503, 'Model unavailable')
    expect(r.type).toBe('model')
    expect(r.message).toContain('Model unavailable')
  })

  it('maps unknown status to unknown error', () => {
    const r = getErrorInfo(418)
    expect(r.type).toBe('unknown')
    expect(r.message).toContain('418')
  })

  it('returns generic message when none provided for 500', () => {
    const r = getErrorInfo(500)
    expect(r.message).toBe('Internal server error.')
  })

  it('returns generic message for 0', () => {
    const r = getErrorInfo(0)
    expect(r.message).toBe('Could not connect to the service.')
  })

  it('maps status 400 to server error (non-retryable)', () => {
    const r = getErrorInfo(400, 'Bad request')
    expect(r.type).toBe('server')
    expect(r.message).toBe('Bad request')
    expect(r.canRetry).toBe(false)
  })

  it('maps status 400 to generic message when none provided', () => {
    const r = getErrorInfo(400)
    expect(r.message).toBe('Invalid request.')
  })

  it('maps status 408 to timeout error', () => {
    const r = getErrorInfo(408, 'Request timed out')
    expect(r.type).toBe('timeout')
    expect(r.message).toBe('Request timed out')
    expect(r.canRetry).toBe(true)
  })

  it('maps status 422 to server error (non-retryable)', () => {
    const r = getErrorInfo(422)
    expect(r.type).toBe('server')
    expect(r.message).toBe('Invalid request format.')
    expect(r.canRetry).toBe(false)
  })

  it('maps status 429 to server error (retryable)', () => {
    const r = getErrorInfo(429)
    expect(r.type).toBe('server')
    expect(r.message).toContain('Too many requests')
    expect(r.canRetry).toBe(true)
  })

  it('maps status 502 to server error', () => {
    const r = getErrorInfo(502)
    expect(r.type).toBe('server')
    expect(r.message).toContain('temporarily unavailable')
    expect(r.canRetry).toBe(true)
  })

  it('maps status 504 to timeout error', () => {
    const r = getErrorInfo(504, 'Gateway timeout')
    expect(r.type).toBe('timeout')
    expect(r.message).toBe('Gateway timeout')
    expect(r.canRetry).toBe(true)
  })

  it('maps status 504 to generic message when none provided', () => {
    const r = getErrorInfo(504)
    expect(r.message).toBe('Generation timed out.')
  })
})

describe('ErrorBanner component', () => {
  it('renders error title and message', () => {
    render(
      <ErrorBanner
        error={{ type: 'network', message: 'Connection lost', canRetry: true }}
        onRetry={() => {}}
        onDismiss={() => {}}
      />
    )
    expect(screen.getByText('Connection failed')).toBeInTheDocument()
    expect(screen.getByText('Connection lost')).toBeInTheDocument()
  })

  it('shows Retry button when canRetry is true', () => {
    render(
      <ErrorBanner
        error={{ type: 'timeout', message: 'Timed out', canRetry: true }}
        onRetry={() => {}}
        onDismiss={() => {}}
      />
    )
    expect(screen.getByText('Retry')).toBeInTheDocument()
  })

  it('hides Retry button when canRetry is false', () => {
    render(
      <ErrorBanner
        error={{ type: 'unknown', message: 'Fatal', canRetry: false }}
        onRetry={() => {}}
        onDismiss={() => {}}
      />
    )
    expect(screen.queryByText('Retry')).not.toBeInTheDocument()
  })

  it('calls onRetry when Retry clicked', () => {
    const onRetry = vi.fn()
    render(
      <ErrorBanner
        error={{ type: 'network', message: 'Error', canRetry: true }}
        onRetry={onRetry}
        onDismiss={() => {}}
      />
    )
    fireEvent.click(screen.getByText('Retry'))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('calls onDismiss when Dismiss clicked', () => {
    const onDismiss = vi.fn()
    render(
      <ErrorBanner
        error={{ type: 'unknown', message: 'Oops', canRetry: false }}
        onRetry={() => {}}
        onDismiss={onDismiss}
      />
    )
    fireEvent.click(screen.getByText('Dismiss'))
    expect(onDismiss).toHaveBeenCalledOnce()
  })

  it('renders correct title for each error type', () => {
    const types = [
      { type: 'network' as const, title: 'Connection failed' },
      { type: 'server' as const, title: 'Service error' },
      { type: 'model' as const, title: 'Model not loaded' },
      { type: 'timeout' as const, title: 'Request timed out' },
      { type: 'unknown' as const, title: 'Something went wrong' },
    ]
    for (const { type, title } of types) {
      const { unmount } = render(
        <ErrorBanner
          error={{ type, message: 'test', canRetry: true }}
          onRetry={() => {}}
          onDismiss={() => {}}
        />
      )
      expect(screen.getByText(title)).toBeInTheDocument()
      unmount()
    }
  })

  it('renders suggestion text for error type', () => {
    render(
      <ErrorBanner
        error={{ type: 'model', message: 'No model', canRetry: true }}
        onRetry={() => {}}
        onDismiss={() => {}}
      />
    )
    expect(screen.getByText(/No model is currently loaded/)).toBeInTheDocument()
  })

  it('has alert role for accessibility', () => {
    render(
      <ErrorBanner
        error={{ type: 'server', message: 'Server error', canRetry: true }}
        onRetry={() => {}}
        onDismiss={() => {}}
      />
    )
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })
})
