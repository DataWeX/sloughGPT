/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { AppRouteHeader, AppRouteHeaderLead } from './AppRouteHeader'

afterEach(cleanup)

describe('AppRouteHeaderLead', () => {
  it('renders title as h1', () => {
    render(<AppRouteHeaderLead title="Dashboard" />)
    const h1 = screen.getByRole('heading', { level: 1 })
    expect(h1).toHaveTextContent('Dashboard')
  })

  it('renders subtitle when provided', () => {
    render(<AppRouteHeaderLead title="Dashboard" subtitle="Welcome back" />)
    expect(screen.getByText('Welcome back')).toBeInTheDocument()
  })

  it('does not render subtitle when absent', () => {
    const { container } = render(<AppRouteHeaderLead title="Dashboard" />)
    const subtitles = container.querySelectorAll('.text-muted-foreground')
    expect(subtitles.length).toBe(0)
  })

  it('renders children', () => {
    render(<AppRouteHeaderLead title="Dashboard"><span data-testid="child">Extra</span></AppRouteHeaderLead>)
    expect(screen.getByTestId('child')).toHaveTextContent('Extra')
  })

  it('handles ReactNode title', () => {
    render(<AppRouteHeaderLead title={<span data-testid="custom-title">Custom</span>} />)
    expect(screen.getByTestId('custom-title')).toHaveTextContent('Custom')
  })
})

describe('AppRouteHeader', () => {
  it('renders left content', () => {
    render(<AppRouteHeader left={<span>Left</span>} />)
    expect(screen.getByText('Left')).toBeInTheDocument()
  })

  it('renders right content when provided', () => {
    render(<AppRouteHeader left={<span>Left</span>} right={<span>Right</span>} />)
    expect(screen.getByText('Right')).toBeInTheDocument()
  })

  it('does not render right container when absent', () => {
    render(<AppRouteHeader left={<span>Left</span>} />)
    expect(screen.queryByText('Right')).not.toBeInTheDocument()
  })

  it('applies sticky class when sticky prop is true', () => {
    const { container } = render(<AppRouteHeader left={<span>L</span>} sticky />)
    expect(container.querySelector('header')).toHaveClass('sticky')
    expect(container.querySelector('header')).toHaveClass('top-0')
  })

  it('does not apply sticky class by default', () => {
    const { container } = render(<AppRouteHeader left={<span>L</span>} />)
    expect(container.querySelector('header')).not.toHaveClass('sticky')
  })

  it('renders as header element', () => {
    const { container } = render(<AppRouteHeader left={<span>L</span>} />)
    expect(container.querySelector('header')).toBeInTheDocument()
  })
})
