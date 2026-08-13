import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Label } from './label'

describe('Label', () => {
  it('renders a label element', () => {
    const html = renderToStaticMarkup(<Label />)
    expect(html).toContain('<label')
    expect(html).toContain('</label>')
  })

  it('displays children', () => {
    const html = renderToStaticMarkup(<Label>Email address</Label>)
    expect(html).toContain('Email address')
  })

  it('applies default variant classes', () => {
    const html = renderToStaticMarkup(<Label>Default</Label>)
    expect(html).toContain('text-sm')
    expect(html).toContain('font-medium')
    expect(html).toContain('text-foreground')
  })

  it('applies muted variant classes', () => {
    const html = renderToStaticMarkup(<Label variant="muted">Muted</Label>)
    expect(html).toContain('text-xs')
    expect(html).toContain('font-medium')
    expect(html).toContain('text-muted-foreground')
    expect(html).not.toContain('text-sm')
  })

  it('applies uppercase variant classes', () => {
    const html = renderToStaticMarkup(<Label variant="uppercase">Uppercase</Label>)
    expect(html).toContain('uppercase')
    expect(html).toContain('tracking-wider')
    expect(html).toContain('text-muted-foreground')
  })

  it('renders a required asterisk with aria-hidden', () => {
    const html = renderToStaticMarkup(<Label required>Name</Label>)
    expect(html).toContain('*')
    expect(html).toContain('aria-hidden="true"')
    expect(html).toContain('text-destructive')
  })

  it('omits the asterisk when not required', () => {
    const html = renderToStaticMarkup(<Label>Name</Label>)
    expect(html).not.toContain('*')
    expect(html).not.toContain('text-destructive')
  })

  it('merges custom className', () => {
    const html = renderToStaticMarkup(<Label className="my-label">Name</Label>)
    expect(html).toContain('my-label')
  })

  it('passes htmlFor to the label', () => {
    const html = renderToStaticMarkup(<Label htmlFor="email">Email</Label>)
    expect(html).toContain('for="email"')
  })

  it('handles empty children', () => {
    const html = renderToStaticMarkup(<Label required />)
    expect(html).toContain('<label')
    expect(html).toContain('*')
  })
})
