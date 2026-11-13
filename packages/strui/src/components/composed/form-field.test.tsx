import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { FormField } from './form-field'

describe('FormField', () => {
  it('renders label linked to the control via htmlFor', () => {
    const html = renderToStaticMarkup(
      <FormField id="name" label="Name">
        <input id="name" />
      </FormField>,
    )
    expect(html).toContain('<label')
    expect(html).toContain('for="name"')
    expect(html).toContain('Name')
  })

  it('renders children', () => {
    const html = renderToStaticMarkup(
      <FormField id="x" label="X">
        <input id="x" />
      </FormField>,
    )
    expect(html).toContain('<input')
  })

  it('shows hint when no error is present', () => {
    const html = renderToStaticMarkup(
      <FormField id="x" label="X" hint="Use at least 8 chars">
        <input id="x" />
      </FormField>,
    )
    expect(html).toContain('Use at least 8 chars')
    expect(html).toContain('id="x-hint"')
  })

  it('shows error and hides hint when error is present', () => {
    const html = renderToStaticMarkup(
      <FormField id="x" label="X" hint="Use at least 8 chars" error="Too short">
        <input id="x" />
      </FormField>,
    )
    expect(html).toContain('Too short')
    expect(html).toContain('id="x-error"')
    expect(html).toContain('role="alert"')
    expect(html).not.toContain('Use at least 8 chars')
    expect(html).not.toContain('id="x-hint"')
  })

  it('renders neither hint nor error when both are absent', () => {
    const html = renderToStaticMarkup(
      <FormField id="x" label="X">
        <input id="x" />
      </FormField>,
    )
    expect(html).not.toContain('-hint')
    expect(html).not.toContain('-error')
  })

  it('passes className to the wrapper', () => {
    const html = renderToStaticMarkup(
      <FormField id="x" label="X" className="my-field">
        <input id="x" />
      </FormField>,
    )
    expect(html).toContain('my-field')
  })
})
