import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import GlobalError from './global-error'

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

describe('GlobalError', () => {
  it('wraps CustomErrorHandler in html and body', () => {
    const { container } = render(<GlobalError error={new Error('fatal')} reset={vi.fn()} />)
    expect(container.querySelector('html')).toBeTruthy()
    expect(container.querySelector('body')).toBeTruthy()
    expect(screen.getByTestId('handler')).toHaveTextContent('fatal|function|Error')
  })

  it('applies background and foreground color classes to body', () => {
    const { container } = render(<GlobalError error={new Error('fatal')} reset={vi.fn()} />)
    const body = container.querySelector('body')
    expect(body?.className).toContain('bg-background')
    expect(body?.className).toContain('text-foreground')
  })

  it('handles TypeError', () => {
    render(<GlobalError error={new TypeError('type err')} reset={vi.fn()} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('type err|function|TypeError')
  })

  it('passes reset as function', () => {
    const reset = vi.fn()
    render(<GlobalError error={new Error('test')} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('test|function|Error')
  })

  it('handles empty error message', () => {
    render(<GlobalError error={new Error('')} reset={vi.fn()} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('|function|Error')
  })

  it('handles custom error name', () => {
    const err = new Error('custom')
    err.name = 'CustomError'
    render(<GlobalError error={err} reset={vi.fn()} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('custom|function|CustomError')
  })
})
