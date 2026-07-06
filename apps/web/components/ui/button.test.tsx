/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Button } from '@sloughgpt/strui'

afterEach(cleanup)

describe('Button', () => {
  it('renders children text', () => {
    render(<Button>Click me</Button>)
    expect(screen.getByText('Click me')).toBeInTheDocument()
  })

  it('renders as button element', () => {
    render(<Button>Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('defaults type to "button"', () => {
    render(<Button>Test</Button>)
    expect(screen.getByRole('button')).toHaveAttribute('type', 'button')
  })

  it('applies type from props', () => {
    render(<Button type="submit">Submit</Button>)
    expect(screen.getByRole('button')).toHaveAttribute('type', 'submit')
  })

  it('applies default variant classes', () => {
    const { container } = render(<Button>Default</Button>)
    expect(container.firstChild).toHaveClass('bg-primary')
    expect(container.firstChild).toHaveClass('text-primary-foreground')
  })

  it('applies secondary variant', () => {
    const { container } = render(<Button variant="secondary">Secondary</Button>)
    expect(container.firstChild).toHaveClass('bg-secondary')
  })

  it('applies ghost variant', () => {
    const { container } = render(<Button variant="ghost">Ghost</Button>)
    expect(container.firstChild).toHaveClass('text-muted-foreground')
  })

  it('applies destructive variant', () => {
    const { container } = render(<Button variant="destructive">Destructive</Button>)
    expect(container.firstChild).toHaveClass('bg-destructive')
  })

  it('applies outline variant', () => {
    const { container } = render(<Button variant="outline">Outline</Button>)
    expect(container.firstChild).toHaveClass('border')
    expect(container.firstChild).toHaveClass('border-border')
  })

  it('applies sm size', () => {
    const { container } = render(<Button size="sm">Small</Button>)
    expect(container.firstChild).toHaveClass('h-9')
    expect(container.firstChild).toHaveClass('text-xs')
  })

  it('applies lg size', () => {
    const { container } = render(<Button size="lg">Large</Button>)
    expect(container.firstChild).toHaveClass('h-11')
  })

  it('applies icon size', () => {
    const { container } = render(<Button size="icon">X</Button>)
    expect(container.firstChild).toHaveClass('h-10')
    expect(container.firstChild).toHaveClass('w-11')
  })

  it('calls onClick when clicked', async () => {
    const onClick = vi.fn()
    const user = userEvent.setup()
    render(<Button onClick={onClick}>Hit</Button>)
    await user.click(screen.getByText('Hit'))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('is disabled when disabled prop is set', () => {
    render(<Button disabled>Disabled</Button>)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('applies custom className', () => {
    const { container } = render(<Button className="my-custom">Custom</Button>)
    expect(container.firstChild).toHaveClass('my-custom')
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Button ref={ref}>Ref</Button>)
    expect(ref).toHaveBeenCalledOnce()
  })

  it('renders with bare variant', () => {
    const { container } = render(<Button variant="bare">Bare</Button>)
    expect(container.firstChild).toHaveClass('text-foreground')
  })

  it('renders with menu variant', () => {
    const { container } = render(<Button variant="menu">Menu</Button>)
    expect(container.firstChild).toHaveClass('text-foreground')
    expect(container.firstChild).toHaveClass('hover:bg-primary/8')
  })

  it('renders with aria-label', () => {
    render(<Button aria-label="Close dialog">X</Button>)
    expect(screen.getByLabelText('Close dialog')).toBeInTheDocument()
  })
})
