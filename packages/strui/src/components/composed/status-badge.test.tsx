import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { StatusBadge } from './status-badge'

describe('StatusBadge', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<StatusBadge>Active</StatusBadge>)
    expect(html).toContain('Active')
  })

  it('applies default tone', () => {
    const html = renderToStaticMarkup(<StatusBadge>Default</StatusBadge>)
    expect(html).toContain('bg-muted')
  })

  it('applies success tone', () => {
    const html = renderToStaticMarkup(<StatusBadge tone="success">OK</StatusBadge>)
    expect(html).toContain('bg-success/15')
  })

  it('applies warning tone', () => {
    const html = renderToStaticMarkup(<StatusBadge tone="warning">Warn</StatusBadge>)
    expect(html).toContain('bg-warning/15')
  })

  it('applies destructive tone', () => {
    const html = renderToStaticMarkup(<StatusBadge tone="destructive">Error</StatusBadge>)
    expect(html).toContain('bg-destructive/15')
  })

  it('applies info tone', () => {
    const html = renderToStaticMarkup(<StatusBadge tone="info">Info</StatusBadge>)
    expect(html).toContain('bg-primary/15')
  })

  it('applies sm size', () => {
    const html = renderToStaticMarkup(<StatusBadge size="sm">SM</StatusBadge>)
    expect(html).toContain('text-[9px]')
  })

  it('applies md size', () => {
    const html = renderToStaticMarkup(<StatusBadge size="md">MD</StatusBadge>)
    expect(html).toContain('text-[10px]')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(<StatusBadge className="custom">X</StatusBadge>)
    expect(html).toContain('custom')
  })
})
