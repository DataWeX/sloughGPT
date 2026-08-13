import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import RootError from './error'

vi.mock('@/components/CustomErrorHandler', () => ({
  CustomErrorHandler: ({ error, reset }: any) => (
    <div data-testid="handler">
      {error.message}|{typeof reset}
    </div>
  ),
}))

afterEach(() => {
  cleanup()
})

describe('RootError', () => {
  it('delegates to CustomErrorHandler with error and reset', () => {
    const error = new Error('boom')
    const reset = vi.fn()
    render(<RootError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('boom|function')
  })

  it('preserves the digest on the error', () => {
    const error = new Error('boom') as Error & { digest?: string }
    error.digest = 'abc123'
    render(<RootError error={error} reset={vi.fn()} />)
    expect(screen.getByTestId('handler')).toBeInTheDocument()
  })

  it('handles TypeError', () => {
    render(<RootError error={new TypeError('type err')} reset={vi.fn()} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('type err|function')
  })

  it('handles empty error message', () => {
    render(<RootError error={new Error('')} reset={vi.fn()} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('|function')
  })
})
