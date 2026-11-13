import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { ColorInput, DEFAULT_THEME_SWATCHES, ThemeColorPicker, ThemeSwatch } from './theme-color-picker'

afterEach(() => {
  cleanup()
})

describe('DEFAULT_THEME_SWATCHES', () => {
  it('is a non-empty array', () => {
    expect(DEFAULT_THEME_SWATCHES.length).toBeGreaterThan(0)
  })

  it('has the expected shape for each swatch', () => {
    for (const swatch of DEFAULT_THEME_SWATCHES) {
      expect(typeof swatch.id).toBe('string')
      expect(typeof swatch.name).toBe('string')
      expect(typeof swatch.color).toBe('string')
    }
  })
})

describe('ThemeSwatch', () => {
  it('renders a button labelled with the swatch name', () => {
    const html = renderToStaticMarkup(
      <ThemeSwatch swatch={{ id: 'pink', name: 'Rose', color: '#d894b4' }} selected={false} onClick={() => {}} />,
    )
    expect(html).toContain('<button')
    expect(html).toContain('aria-label="Rose theme"')
    expect(html).toContain('aria-pressed="false"')
  })

  it('sets aria-pressed when selected', () => {
    const html = renderToStaticMarkup(
      <ThemeSwatch swatch={{ id: 'pink', name: 'Rose', color: '#d894b4' }} selected onClick={() => {}} />,
    )
    expect(html).toContain('aria-pressed="true"')
  })

  it('fires onClick when clicked', () => {
    const onClick = vi.fn()
    render(
      <ThemeSwatch swatch={{ id: 'pink', name: 'Rose', color: '#d894b4' }} selected={false} onClick={onClick} />,
    )
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('shows the label when showLabel is set', () => {
    const html = renderToStaticMarkup(
      <ThemeSwatch swatch={{ id: 'pink', name: 'Rose', color: '#d894b4' }} selected={false} onClick={() => {}} showLabel />,
    )
    expect(html).toContain('Rose')
  })
})

describe('ThemeColorPicker', () => {
  it('renders swatches and fires onChange with the swatch color', () => {
    const onChange = vi.fn()
    render(<ThemeColorPicker value="#8b7bc4" onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Peach theme'))
    expect(onChange).toHaveBeenCalledWith('#e8a86c')
  })

  it('renders the custom color button by default', () => {
    render(<ThemeColorPicker value="#8b7bc4" onChange={() => {}} />)
    expect(screen.getByLabelText('Custom color')).toBeTruthy()
  })

  it('hides the custom color button when showCustomInput is false', () => {
    const html = renderToStaticMarkup(
      <ThemeColorPicker value="#8b7bc4" onChange={() => {}} showCustomInput={false} />,
    )
    expect(html).not.toContain('Custom color')
  })

  it('renders a custom label', () => {
    const html = renderToStaticMarkup(<ThemeColorPicker value="#8b7bc4" onChange={() => {}} label="Theme accent" />)
    expect(html).toContain('Theme accent')
  })

  it('renders swatches from the swatches prop', () => {
    const swatches = [{ id: 'x', name: 'Slate', color: '#111111' }]
    const html = renderToStaticMarkup(<ThemeColorPicker value="#111111" onChange={() => {}} swatches={swatches} />)
    expect(html).toContain('Slate theme')
  })

  it('passes className to the wrapper', () => {
    const html = renderToStaticMarkup(<ThemeColorPicker value="#8b7bc4" onChange={() => {}} className="my-picker" />)
    expect(html).toContain('my-picker')
  })
})

describe('ColorInput', () => {
  it('renders a color input and a text input', () => {
    const html = renderToStaticMarkup(<ColorInput label="Accent" />)
    expect(html).toContain('type="color"')
    expect(html).toContain('type="text"')
  })

  it('renders the label when provided', () => {
    const html = renderToStaticMarkup(<ColorInput label="Accent" />)
    expect(html).toContain('Accent')
  })

  it('passes props to the inputs', () => {
    const onChange = vi.fn()
    render(<ColorInput label="Accent" value="#123456" onChange={onChange} />)
    const textInput = screen.getAllByRole('textbox')[0]
    expect(textInput.getAttribute('value')).toBe('#123456')
    fireEvent.change(textInput, { target: { value: '#abcdef' } })
    expect(onChange).toHaveBeenCalled()
  })

  it('passes className to the wrapper', () => {
    const html = renderToStaticMarkup(<ColorInput className="my-input" />)
    expect(html).toContain('my-input')
  })
})
