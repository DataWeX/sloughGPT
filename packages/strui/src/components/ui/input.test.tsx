import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Input, SearchInput, inputFieldClassName, sanitizeInputType } from './input'

describe('sanitizeInputType', () => {
  it('keeps safe text-like types', () => {
    for (const t of ['text', 'email', 'password', 'number', 'search', 'tel', 'url', 'date', 'time']) {
      expect(sanitizeInputType(t)).toBe(t)
    }
  })

  it('coerces control types to text', () => {
    for (const t of ['file', 'checkbox', 'radio', 'button', 'hidden', 'submit', 'reset', 'image']) {
      expect(sanitizeInputType(t)).toBe('text')
    }
  })

  it('coerces unknown and empty types to text', () => {
    expect(sanitizeInputType('banana')).toBe('text')
    expect(sanitizeInputType(undefined)).toBe('text')
  })
})

describe('Input', () => {
  it('renders input element', () => {
    const html = renderToStaticMarkup(<Input />)
    expect(html).toContain('<input')
  })

  it('uses border-border not border-input', () => {
    expect(inputFieldClassName).toContain('border-border')
    expect(inputFieldClassName).not.toContain('border-input')
  })

  it('hover uses border-border/80 not border-primary/50', () => {
    expect(inputFieldClassName).toContain('hover:border-border/80')
    expect(inputFieldClassName).not.toContain('hover:border-primary/50')
  })

  it('focus ring uses ring-primary/30', () => {
    expect(inputFieldClassName).toContain('ring-primary/30')
  })

  it('disabled uses opacity-40', () => {
    expect(inputFieldClassName).toContain('disabled:opacity-40')
    expect(inputFieldClassName).not.toContain('disabled:opacity-50')
  })

  it('disabled includes bg-muted', () => {
    expect(inputFieldClassName).toContain('disabled:bg-muted')
  })

  it('placeholder uses /50 opacity', () => {
    expect(inputFieldClassName).toContain('placeholder:text-muted-foreground/50')
  })

  it('applies error classes when error prop is true', () => {
    const html = renderToStaticMarkup(<Input error />)
    expect(html).toContain('border-destructive')
    expect(html).toContain('ring-destructive/20')
  })

  it('renders with leftIcon', () => {
    const html = renderToStaticMarkup(
      <Input leftIcon={<span data-icon="search">S</span>} />
    )
    expect(html).toContain('data-icon="search"')
    expect(html).toContain('pl-9')
  })

  it('renders with rightElement', () => {
    const html = renderToStaticMarkup(
      <Input rightElement={<span data-action="clear">X</span>} />
    )
    expect(html).toContain('data-action="clear"')
    expect(html).toContain('pr-9')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Input className="my-class" />)
    expect(html).toContain('my-class')
  })

  it('passes type prop', () => {
    const html = renderToStaticMarkup(<Input type="password" />)
    expect(html).toContain('type="password"')
  })

  it('sanitizes unsupported type to text', () => {
    const html = renderToStaticMarkup(<Input type="file" />)
    expect(html).toContain('type="text"')
    expect(html).not.toContain('type="file"')
  })

  it('defaults type to text', () => {
    const html = renderToStaticMarkup(<Input />)
    expect(html).toContain('type="text"')
  })

  it('sets aria-invalid when error', () => {
    const html = renderToStaticMarkup(<Input error aria-describedby="x-error" />)
    expect(html).toContain('aria-invalid="true"')
  })

  it('omits aria-invalid when not in error', () => {
    const html = renderToStaticMarkup(<Input />)
    expect(html).not.toContain('aria-invalid')
  })

  it('passes placeholder prop', () => {
    const html = renderToStaticMarkup(<Input placeholder="Enter text" />)
    expect(html).toContain('placeholder="Enter text"')
  })
})

describe('SearchInput', () => {
  it('renders search input', () => {
    const html = renderToStaticMarkup(<SearchInput />)
    expect(html).toContain('type="search"')
  })

  it('renders search icon SVG', () => {
    const html = renderToStaticMarkup(<SearchInput />)
    expect(html).toContain('M21 21l-6-6m2-5a7 7 0 11-14 0')
  })

  it('shows clear button when value is provided', () => {
    const html = renderToStaticMarkup(<SearchInput value="test" onChange={() => {}} />)
    expect(html).toContain('aria-label="Clear search"')
  })

  it('hides clear button when value is empty', () => {
    const html = renderToStaticMarkup(<SearchInput value="" onChange={() => {}} />)
    expect(html).not.toContain('Clear search')
  })

  it('defaults aria-label to Search', () => {
    const html = renderToStaticMarkup(<SearchInput />)
    expect(html).toContain('aria-label="Search"')
  })

  it('respects a provided aria-label', () => {
    const html = renderToStaticMarkup(<SearchInput aria-label="Find models" />)
    expect(html).toContain('aria-label="Find models"')
    expect(html).not.toContain('aria-label="Search"')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<SearchInput className="custom" />)
    expect(html).toContain('custom')
  })
})
