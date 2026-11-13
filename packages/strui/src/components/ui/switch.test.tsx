import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Switch } from './switch'

describe('Switch', () => {
  it('renders as switch role', () => {
    const html = renderToStaticMarkup(<Switch />)
    expect(html).toContain('role="switch"')
  })

  it('defaults to unchecked', () => {
    const html = renderToStaticMarkup(<Switch />)
    expect(html).toContain('aria-checked="false"')
  })

  it('renders checked when defaultChecked is true', () => {
    const html = renderToStaticMarkup(<Switch defaultChecked />)
    expect(html).toContain('aria-checked="true"')
  })

  it('applies bg-primary when checked', () => {
    const html = renderToStaticMarkup(<Switch defaultChecked />)
    expect(html).toContain('bg-primary')
  })

  it('applies bg-muted when unchecked', () => {
    const html = renderToStaticMarkup(<Switch />)
    expect(html).toContain('bg-muted')
  })

  it('includes hover brightness', () => {
    const html = renderToStaticMarkup(<Switch />)
    expect(html).toContain('hover:brightness-110')
  })

  it('includes focus ring-ring', () => {
    const html = renderToStaticMarkup(<Switch />)
    expect(html).toContain('ring-ring')
  })

  it('renders disabled when disabled prop is true', () => {
    const html = renderToStaticMarkup(<Switch disabled />)
    expect(html).toContain('disabled')
    expect(html).toContain('opacity-40')
    expect(html).not.toContain('disabled:opacity-50')
  })

  it('renders label text when provided', () => {
    const html = renderToStaticMarkup(<Switch label="Enable feature" />)
    expect(html).toContain('Enable feature')
  })

  it('has hidden checkbox input for form submission', () => {
    const html = renderToStaticMarkup(<Switch />)
    expect(html).toContain('type="checkbox"')
    expect(html).toContain('sr-only')
  })

  it('applies sm size classes', () => {
    const html = renderToStaticMarkup(<Switch size="sm" />)
    expect(html).toContain('h-4')
    expect(html).toContain('w-7')
  })

  it('applies default size classes', () => {
    const html = renderToStaticMarkup(<Switch size="default" />)
    expect(html).toContain('h-6')
    expect(html).toContain('w-11')
  })
})
