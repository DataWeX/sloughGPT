import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Chip, Chips, TagInput } from './tags'

describe('Chip', () => {
  it('renders label', () => {
    const html = renderToStaticMarkup(<Chip label="Tag" />)
    expect(html).toContain('Tag')
  })

  it('renders as span when no onClick', () => {
    const html = renderToStaticMarkup(<Chip label="Static" />)
    expect(html).toContain('<span')
  })

  it('renders as button when onClick is provided', () => {
    const html = renderToStaticMarkup(<Chip label="Clickable" onClick={() => {}} />)
    expect(html).toContain('<button')
    expect(html).toContain('type="button"')
  })

  it('default unselected uses secondary colors', () => {
    const html = renderToStaticMarkup(<Chip label="Tag" />)
    expect(html).toContain('bg-secondary')
    expect(html).toContain('text-secondary-foreground')
  })

  it('default selected uses primary', () => {
    const html = renderToStaticMarkup(<Chip label="Tag" selected />)
    expect(html).toContain('bg-primary')
    expect(html).toContain('text-primary-foreground')
  })

  it('success variant unselected', () => {
    const html = renderToStaticMarkup(<Chip label="OK" variant="success" />)
    expect(html).toContain('bg-success/15')
    expect(html).toContain('text-success')
  })

  it('warning variant unselected', () => {
    const html = renderToStaticMarkup(<Chip label="Caution" variant="warning" />)
    expect(html).toContain('bg-warning/15')
    expect(html).toContain('text-warning')
  })

  it('error variant unselected', () => {
    const html = renderToStaticMarkup(<Chip label="Fail" variant="error" />)
    expect(html).toContain('bg-destructive/15')
    expect(html).toContain('text-destructive')
  })

  it('shows remove button when removable', () => {
    const html = renderToStaticMarkup(<Chip label="Tag" removable onRemove={() => {}} />)
    expect(html).toContain('aria-label="Remove Tag"')
  })

  it('hides remove button when not removable', () => {
    const html = renderToStaticMarkup(<Chip label="Tag" />)
    expect(html).not.toContain('Remove')
  })

  it('disabled has opacity-40', () => {
    const html = renderToStaticMarkup(<Chip label="Tag" disabled onClick={() => {}} />)
    expect(html).toContain('opacity-40')
    expect(html).not.toContain('opacity-50')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Chip label="X" className="custom" />)
    expect(html).toContain('custom')
  })

  it('has rounded-full', () => {
    const html = renderToStaticMarkup(<Chip label="X" />)
    expect(html).toContain('rounded-full')
  })

  it('sets aria-pressed when clickable', () => {
    const html = renderToStaticMarkup(<Chip label="X" onClick={() => {}} selected />)
    expect(html).toContain('aria-pressed="true"')
  })

  it('does not set aria-pressed when static', () => {
    const html = renderToStaticMarkup(<Chip label="X" />)
    expect(html).not.toContain('aria-pressed')
  })
})

describe('Chips', () => {
  const options = [
    { value: 'a', label: 'Alpha' },
    { value: 'b', label: 'Beta' },
    { value: 'c', label: 'Gamma' },
  ]

  it('renders all options', () => {
    const html = renderToStaticMarkup(<Chips value={[]} onChange={() => {}} options={options} />)
    expect(html).toContain('Alpha')
    expect(html).toContain('Beta')
    expect(html).toContain('Gamma')
  })

  it('has group role', () => {
    const html = renderToStaticMarkup(<Chips value={[]} onChange={() => {}} options={options} />)
    expect(html).toContain('role="group"')
  })

  it('selects matching values', () => {
    const html = renderToStaticMarkup(<Chips value={['a']} onChange={() => {}} options={options} />)
    expect(html).toContain('aria-pressed="true"')
  })

  it('disables chips when max is reached', () => {
    const html = renderToStaticMarkup(
      <Chips value={['a', 'b']} onChange={() => {}} options={options} max={2} />
    )
    expect(html).toContain('disabled')
  })
})

describe('TagInput', () => {
  it('renders input', () => {
    const html = renderToStaticMarkup(<TagInput value={[]} onChange={() => {}} />)
    expect(html).toContain('<input')
  })

  it('renders placeholder', () => {
    const html = renderToStaticMarkup(<TagInput value={[]} onChange={() => {}} placeholder="Add tags" />)
    expect(html).toContain('Add tags')
  })

  it('renders existing tags as Chips', () => {
    const html = renderToStaticMarkup(<TagInput value={['react', 'ts']} onChange={() => {}} />)
    expect(html).toContain('react')
    expect(html).toContain('ts')
  })

  it('has focus-within ring', () => {
    const html = renderToStaticMarkup(<TagInput value={[]} onChange={() => {}} />)
    expect(html).toContain('focus-within:ring-primary/40')
  })

  it('disabled has opacity-40', () => {
    const html = renderToStaticMarkup(<TagInput value={[]} onChange={() => {}} disabled />)
    expect(html).toContain('opacity-40')
  })
})
