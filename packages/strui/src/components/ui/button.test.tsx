import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Button, buttonVariants } from './button'

describe('Button', () => {
  it('renders children text', () => {
    const html = renderToStaticMarkup(<Button>Click me</Button>)
    expect(html).toContain('Click me')
  })

  it('renders as button element', () => {
    const html = renderToStaticMarkup(<Button>OK</Button>)
    expect(html).toContain('<button')
    expect(html).toContain('type="button"')
  })

  it('applies default variant classes', () => {
    const cls = buttonVariants({ variant: 'default' })
    expect(cls).toContain('bg-primary')
    expect(cls).toContain('text-primary-foreground')
  })

  it('applies destructive variant classes', () => {
    const cls = buttonVariants({ variant: 'destructive' })
    expect(cls).toContain('bg-destructive')
    expect(cls).toContain('text-destructive-foreground')
  })

  it('applies secondary variant classes', () => {
    const cls = buttonVariants({ variant: 'secondary' })
    expect(cls).toContain('bg-secondary')
    expect(cls).toContain('text-secondary-foreground')
  })

  it('applies ghost variant classes', () => {
    const cls = buttonVariants({ variant: 'ghost' })
    expect(cls).toContain('text-muted-foreground')
  })

  it('includes disabled opacity-40', () => {
    const cls = buttonVariants()
    expect(cls).toContain('disabled:opacity-40')
  })

  it('includes focus ring-ring', () => {
    const cls = buttonVariants()
    expect(cls).toContain('ring-ring')
  })

  it('includes active scale', () => {
    const cls = buttonVariants()
    expect(cls).toContain('active:scale-[0.98]')
  })

  it('applies size sm classes', () => {
    const cls = buttonVariants({ size: 'sm' })
    expect(cls).toContain('h-9')
  })

  it('applies size lg classes', () => {
    const cls = buttonVariants({ size: 'lg' })
    expect(cls).toContain('h-11')
  })

  it('renders disabled when disabled prop is true', () => {
    const html = renderToStaticMarkup(<Button disabled>OK</Button>)
    expect(html).toContain('disabled')
  })

  it('renders disabled when loading', () => {
    const html = renderToStaticMarkup(<Button loading>Saving</Button>)
    expect(html).toContain('disabled')
    expect(html).toContain('aria-busy="true"')
  })

  it('shows spinner SVG when loading', () => {
    const html = renderToStaticMarkup(<Button loading>Saving</Button>)
    expect(html).toContain('animate-spin')
  })

  it('shows loadingText when provided', () => {
    const html = renderToStaticMarkup(<Button loading loadingText="Please wait">Save</Button>)
    expect(html).toContain('Please wait')
  })

  it('renders leftIcon', () => {
    const html = renderToStaticMarkup(
      <Button leftIcon={<span data-icon="left">L</span>}>Text</Button>
    )
    expect(html).toContain('data-icon="left"')
  })

  it('renders rightIcon', () => {
    const html = renderToStaticMarkup(
      <Button rightIcon={<span data-icon="right">R</span>}>Text</Button>
    )
    expect(html).toContain('data-icon="right"')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Button className="my-custom">X</Button>)
    expect(html).toContain('my-custom')
  })

  it('hover uses bg-primary/90 not opacity-90', () => {
    const cls = buttonVariants({ variant: 'default' })
    expect(cls).toContain('hover:bg-primary/90')
    expect(cls).not.toContain('hover:opacity-90')
  })

  it('destructive hover uses bg-destructive/90 not opacity-90', () => {
    const cls = buttonVariants({ variant: 'destructive' })
    expect(cls).toContain('hover:bg-destructive/90')
    expect(cls).not.toContain('hover:opacity-90')
  })
})
