import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import RouteError from './error'

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

describe('AppRouteError', () => {
  it('delegates to CustomErrorHandler with error and reset', () => {
    const error = new Error('route failed')
    const reset = vi.fn()
    render(<RouteError error={error} reset={reset} />)
    expect(screen.getByTestId('handler')).toHaveTextContent('route failed|function')
  })
})
