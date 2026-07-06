/**
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Badge } from '@sloughgpt/strui'

afterEach(cleanup)

describe('Badge', () => {
  it('renders text content', () => {
    render(<Badge>Active</Badge>)
    expect(screen.getByText('Active')).toBeInTheDocument()
  })

  it('applies default variant classes', () => {
    const { container } = render(<Badge>Default</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('bg-primary')
    expect(el.className).toContain('text-primary-foreground')
  })

  it('applies secondary variant', () => {
    const { container } = render(<Badge variant="secondary">Secondary</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('bg-secondary')
    expect(el.className).toContain('text-secondary-foreground')
  })

  it('applies destructive variant', () => {
    const { container } = render(<Badge variant="destructive">Remove</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('bg-destructive')
  })

  it('applies outline variant', () => {
    const { container } = render(<Badge variant="outline">Outline</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('border-border')
  })

  it('applies success variant', () => {
    const { container } = render(<Badge variant="success">Success</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('text-success')
  })

  it('applies warning variant', () => {
    const { container } = render(<Badge variant="warning">Warning</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('text-warning')
  })

  it('applies sm size', () => {
    const { container } = render(<Badge size="sm">Small</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('text-[10px]')
  })

  it('applies lg size', () => {
    const { container } = render(<Badge size="lg">Large</Badge>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('text-sm')
  })

  it('renders as div by default', () => {
    const { container } = render(<Badge>Test</Badge>)
    expect(container.firstChild?.nodeName).toBe('DIV')
  })

  it('applies custom className', () => {
    const { container } = render(<Badge className="custom-class">Test</Badge>)
    expect(container.firstChild as HTMLElement).toHaveClass('custom-class')
  })
})
