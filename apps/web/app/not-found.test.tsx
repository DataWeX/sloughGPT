import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import NotFound from './not-found'

vi.mock('next/link', () => ({
  default: ({ children, href, className, ...rest }: any) => (
    <a href={href} className={className} {...rest}>
      {children}
    </a>
  ),
}))

afterEach(() => {
  cleanup()
})

describe('NotFound', () => {
  it('renders the 404 heading and description', () => {
    render(<NotFound />)
    expect(screen.getByText('Page not found')).toBeInTheDocument()
    expect(
      screen.getByText(/doesn't exist or has been moved/),
    ).toBeInTheDocument()
  })

  it('links Home to /', () => {
    render(<NotFound />)
    expect(screen.getByRole('link', { name: 'Home' })).toHaveAttribute('href', '/')
  })

  it('links Chat to /chat', () => {
    render(<NotFound />)
    expect(screen.getByRole('link', { name: 'Chat' })).toHaveAttribute('href', '/chat')
  })
})
