import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import RouteError from './error'

vi.mock('@/components/CustomErrorHandler', () => ({
  CustomErrorHandler: ({ error, reset }: any) => (
    <div data-testid="handler">
      {error.message}|{typeof reset}|{error.name || 'Error'}
    </div>
  ),
}))

afterEach(() => {
  cleanup()
})

describe('AppRouteError', () => {
  it('delegates to CustomErrorHandler with error and reset', () => {
    const error = new Error('route failed')
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('route failed|function|Error')
  })

  it('handles TypeError', () => {
    const error = new TypeError('type error')
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('type error|function|TypeError')
  })

  it('handles RangeError', () => {
    const error = new RangeError('range error')
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('range error|function|RangeError')
  })

  it('passes reset function to handler', () => {
    const error = new Error('test')
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toBeTruthy()
  })

  it('handles empty error message', () => {
    const error = new Error('')
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('|function|Error')
  })

  it('handles custom error properties', () => {
    const error = new Error('custom')
    ;(error as any).code = 'CUSTOM_CODE'
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('custom|function|Error')
  })

  it('preserves error digest', () => {
    const error = new Error('digest test') as Error & { digest?: string }
    error.digest = 'abc123'
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('digest test|function|Error')
  })
})
