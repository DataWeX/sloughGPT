import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi, afterEach } from 'vitest'

import { ToggleGroup, ToggleGroupItem } from './toggle-group'

afterEach(() => cleanup())

describe('ToggleGroup', () => {
  it('renders a radiogroup with radio items in single mode', () => {
    const html = renderToStaticMarkup(
      <ToggleGroup type="single">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(html).toContain('role="radiogroup"')
    expect(html).toContain('role="radio"')
    expect(html).toContain('data-state="off"')
    expect(html).toContain('aria-checked="false"')
  })

  it('renders a group with checkbox items in multiple mode', () => {
    const html = renderToStaticMarkup(
      <ToggleGroup type="multiple">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(html).toContain('role="group"')
    expect(html).toContain('role="checkbox"')
  })

  it('selects the defaultValue item in single mode', () => {
    render(
      <ToggleGroup type="single" defaultValue="b">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(screen.getByRole('radio', { name: 'B' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('radio', { name: 'A' }).getAttribute('aria-checked')).toBe('false')
  })

  it('selects no item by default', () => {
    render(
      <ToggleGroup type="single">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(screen.getByRole('radio', { name: 'A' }).getAttribute('aria-checked')).toBe('false')
    expect(screen.getByRole('radio', { name: 'A' }).getAttribute('data-state')).toBe('off')
  })

  it('updates the selection on click in single mode', () => {
    render(
      <ToggleGroup type="single" defaultValue="a">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    fireEvent.click(screen.getByRole('radio', { name: 'B' }))
    expect(screen.getByRole('radio', { name: 'B' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('radio', { name: 'A' }).getAttribute('aria-checked')).toBe('false')
  })

  it('deselects the active item when clicked again in single mode', () => {
    render(
      <ToggleGroup type="single" defaultValue="a">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    fireEvent.click(screen.getByRole('radio', { name: 'A' }))
    expect(screen.getByRole('radio', { name: 'A' }).getAttribute('aria-checked')).toBe('false')
  })

  it('calls onValueChange with the single value', () => {
    const onValueChange = vi.fn()
    render(
      <ToggleGroup type="single" onValueChange={onValueChange}>
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    fireEvent.click(screen.getByRole('radio', { name: 'B' }))
    expect(onValueChange).toHaveBeenCalledWith('b')
  })

  it('respects a controlled single value', () => {
    const onValueChange = vi.fn()
    render(
      <ToggleGroup type="single" value="a" onValueChange={onValueChange}>
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    fireEvent.click(screen.getByRole('radio', { name: 'B' }))
    expect(onValueChange).toHaveBeenCalledWith('b')
    expect(screen.getByRole('radio', { name: 'A' }).getAttribute('aria-checked')).toBe('true')
  })

  it('toggles items on and off in multiple mode', () => {
    render(
      <ToggleGroup type="multiple">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    fireEvent.click(screen.getByRole('checkbox', { name: 'A' }))
    expect(screen.getByRole('checkbox', { name: 'A' }).getAttribute('aria-checked')).toBe('true')
    fireEvent.click(screen.getByRole('checkbox', { name: 'B' }))
    expect(screen.getByRole('checkbox', { name: 'B' }).getAttribute('aria-checked')).toBe('true')
    fireEvent.click(screen.getByRole('checkbox', { name: 'A' }))
    expect(screen.getByRole('checkbox', { name: 'A' }).getAttribute('aria-checked')).toBe('false')
    expect(screen.getByRole('checkbox', { name: 'B' }).getAttribute('aria-checked')).toBe('true')
  })

  it('calls onValueChange with the array in multiple mode', () => {
    const onValueChange = vi.fn()
    render(
      <ToggleGroup type="multiple" onValueChange={onValueChange}>
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    fireEvent.click(screen.getByRole('checkbox', { name: 'A' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'B' }))
    expect(onValueChange).toHaveBeenLastCalledWith(['a', 'b'])
  })

  it('respects a controlled multiple value', () => {
    render(
      <ToggleGroup type="multiple" value={['a']}>
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(screen.getByRole('checkbox', { name: 'A' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('checkbox', { name: 'B' }).getAttribute('aria-checked')).toBe('false')
  })

  it('disables all items when the group is disabled', () => {
    const { container } = render(
      <ToggleGroup type="single" disabled>
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(container.querySelector<HTMLElement>('[role="radiogroup"]')!.getAttribute('aria-disabled')).toBe('true')
    const item = screen.getByRole('radio', { name: 'A' })
    expect(item.getAttribute('disabled')).not.toBeNull()
    expect(item.getAttribute('aria-disabled')).toBe('true')
  })

  it('supports item-level disabled', () => {
    render(
      <ToggleGroup type="single">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b" disabled>
          B
        </ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(screen.getByRole('radio', { name: 'B' }).getAttribute('disabled')).not.toBeNull()
    expect(screen.getByRole('radio', { name: 'A' }).getAttribute('disabled')).toBeNull()
  })

  it('does not fire onValueChange when disabled', () => {
    const onValueChange = vi.fn()
    render(
      <ToggleGroup type="single" disabled onValueChange={onValueChange}>
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    fireEvent.click(screen.getByRole('radio', { name: 'A' }))
    expect(onValueChange).not.toHaveBeenCalled()
  })

  it('applies the default variant container styles', () => {
    const html = renderToStaticMarkup(
      <ToggleGroup type="single">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(html).toContain('rounded-lg border border-border bg-muted/50')
  })

  it('applies the outline variant container styles', () => {
    const html = renderToStaticMarkup(
      <ToggleGroup type="single" variant="outline">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(html).toContain('inline-flex items-center justify-center gap-1')
  })

  it('applies the pills variant container styles', () => {
    const html = renderToStaticMarkup(
      <ToggleGroup type="single" variant="pills">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(html).toContain('flex-wrap')
  })

  it('merges custom className on the container', () => {
    const { container } = render(
      <ToggleGroup type="single" className="my-group">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
      </ToggleGroup>,
    )
    expect(container.querySelector<HTMLElement>('[role="radiogroup"]')!.classList.contains('my-group')).toBe(true)
  })

  it('throws when an item is rendered outside a group', () => {
    expect(() =>
      renderToStaticMarkup(<ToggleGroupItem value="a">X</ToggleGroupItem>),
    ).toThrow('ToggleGroupItem must be used within <ToggleGroup>')
  })
})
