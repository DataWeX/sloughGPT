import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Badge, badgeVariants } from './badge'

describe('Badge', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<Badge>New</Badge>)
    expect(html).toContain('New')
  })

  it('renders label prop over children', () => {
    const html = renderToStaticMarkup(<Badge label="Labeled">Child</Badge>)
    expect(html).toContain('Labeled')
    expect(html).not.toContain('Child')
  })

  it('default variant is secondary (neutral)', () => {
    const cls = badgeVariants({ variant: 'default' })
    expect(cls).toContain('bg-secondary')
    expect(cls).toContain('text-secondary-foreground')
  })

  it('primary variant uses bg-primary/15', () => {
    const cls = badgeVariants({ variant: 'primary' })
    expect(cls).toContain('bg-primary/15')
    expect(cls).toContain('text-primary')
  })

  it('success variant uses bg-success/15', () => {
    const cls = badgeVariants({ variant: 'success' })
    expect(cls).toContain('bg-success/15')
    expect(cls).toContain('text-success')
  })

  it('warning variant uses bg-warning/15', () => {
    const cls = badgeVariants({ variant: 'warning' })
    expect(cls).toContain('bg-warning/15')
    expect(cls).toContain('text-warning')
  })

  it('error variant uses bg-destructive/15', () => {
    const cls = badgeVariants({ variant: 'error' })
    expect(cls).toContain('bg-destructive/15')
    expect(cls).toContain('text-destructive')
  })

  it('destructive variant uses solid destructive', () => {
    const cls = badgeVariants({ variant: 'destructive' })
    expect(cls).toContain('bg-destructive')
    expect(cls).toContain('text-destructive-foreground')
  })

  it('outline variant has border', () => {
    const cls = badgeVariants({ variant: 'outline' })
    expect(cls).toContain('border-border')
  })

  it('includes focus ring-ring', () => {
    const cls = badgeVariants()
    expect(cls).toContain('ring-ring')
  })

  it('is rounded-full', () => {
    const cls = badgeVariants()
    expect(cls).toContain('rounded-full')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Badge className="custom">X</Badge>)
    expect(html).toContain('custom')
  })

  it('applies sm size', () => {
    const cls = badgeVariants({ size: 'sm' })
    expect(cls).toContain('text-[10px]')
  })

  it('applies lg size', () => {
    const cls = badgeVariants({ size: 'lg' })
    expect(cls).toContain('text-sm')
  })
})
