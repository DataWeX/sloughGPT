import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import GlobalError from './global-error'

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

describe('GlobalError', () => {
  it('wraps CustomErrorHandler in html and body', () => {
    const { container } = render(<GlobalError error={new Error('fatal')} reset={vi.fn()} />)
    expect(container.querySelector('html')).toBeTruthy()
    expect(container.querySelector('body')).toBeTruthy()
    expect(screen.getByTestId('handler')).toHaveTextContent('fatal|function')
  })

  it('applies background and foreground color classes to body', () => {
    const { container } = render(<GlobalError error={new Error('fatal')} reset={vi.fn()} />)
    const body = container.querySelector('body')
    expect(body?.className).toContain('bg-background')
    expect(body?.className).toContain('text-foreground')
  })
})
