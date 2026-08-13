import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { Select, SelectTrigger, SelectContent, SelectItem } from './select'

function stubLayout(rect: Partial<DOMRect>, height: number, width: number) {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0, x: 0, y: 0, toJSON: () => ({}),
    ...rect,
  } as DOMRect)
  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', { configurable: true, get: () => height })
  Object.defineProperty(HTMLElement.prototype, 'offsetWidth', { configurable: true, get: () => width })
}

function renderSelect() {
  render(
    <Select defaultValue="a">
      <SelectTrigger data-testid="trigger">Pick</SelectTrigger>
      <SelectContent>
        <SelectItem value="a">Alpha</SelectItem>
        <SelectItem value="b">Beta</SelectItem>
      </SelectContent>
    </Select>,
  )
}

describe('Select', () => {
  beforeEach(() => {
    stubLayout({ top: 100, bottom: 140, left: 30, width: 200, height: 40 }, 200, 300)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    delete (HTMLElement.prototype as unknown as Record<string, unknown>).offsetHeight
    delete (HTMLElement.prototype as unknown as Record<string, unknown>).offsetWidth
  })

  it('renders trigger with combobox semantics', () => {
    renderSelect()
    const trigger = screen.getByTestId('trigger')
    expect(trigger.getAttribute('role')).toBe('combobox')
    expect(trigger.getAttribute('aria-haspopup')).toBe('listbox')
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(trigger.getAttribute('aria-controls')).toBeTruthy()
  })

  it('opens a listbox anchored below the trigger', () => {
    renderSelect()
    fireEvent.click(screen.getByTestId('trigger'))

    const listbox = screen.getByRole('listbox')
    expect(listbox).toBeTruthy()
    expect(listbox.getAttribute('role')).toBe('listbox')
    expect(listbox.classList.contains('fixed')).toBe(true)
    expect(listbox.style.top).toBe('146px')
    expect(listbox.style.left).toBe('30px')
    expect(listbox.style.minWidth).toBe('200px')
  })

  it('links the listbox to the trigger via aria-controls', () => {
    renderSelect()
    const trigger = screen.getByTestId('trigger')
    fireEvent.click(trigger)
    const listbox = screen.getByRole('listbox')
    expect(listbox.id).toBe(trigger.getAttribute('aria-controls'))
  })

  it('flips above the trigger when there is no room below', () => {
    stubLayout({ top: 600, bottom: 640, left: 30, width: 200, height: 40 }, 400, 300)
    renderSelect()
    fireEvent.click(screen.getByTestId('trigger'))

    const listbox = screen.getByRole('listbox')
    expect(listbox.style.top).toBe('194px')
  })

  it('closes on Escape', () => {
    renderSelect()
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('listbox')).toBeTruthy()

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('selects an item on click and closes', () => {
    const onValueChange = vi.fn()
    render(
      <Select onValueChange={onValueChange}>
        <SelectTrigger data-testid="trigger">Pick</SelectTrigger>
        <SelectContent>
          <SelectItem value="a">Alpha</SelectItem>
          <SelectItem value="b">Beta</SelectItem>
        </SelectContent>
      </Select>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    fireEvent.click(screen.getByText('Beta'))

    expect(screen.queryByRole('listbox')).toBeNull()
    expect(onValueChange).toHaveBeenCalledWith('b')
  })
})
