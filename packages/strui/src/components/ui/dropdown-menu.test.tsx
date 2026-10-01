import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
} from './dropdown-menu'

function renderMenu(props: { onOpenChange?: (open: boolean) => void } = {}) {
  return render(
    <DropdownMenu {...props}>
      <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuItem data-testid="item-a">Item A</DropdownMenuItem>
        <DropdownMenuItem data-testid="item-b">Item B</DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuCheckboxItem checked={false} data-testid="toggle">
          Toggle option
        </DropdownMenuCheckboxItem>
      </DropdownMenuContent>
    </DropdownMenu>,
  )
}

describe('DropdownMenu', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      cb(0)
      return 0
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('renders a trigger with menu semantics', () => {
    renderMenu()
    const trigger = screen.getByTestId('trigger')
    expect(trigger.getAttribute('type')).toBe('button')
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu')
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
  })

  it('does not render content while closed', () => {
    renderMenu()
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('opens on trigger click and renders menu items', () => {
    renderMenu()
    fireEvent.click(screen.getByTestId('trigger'))
    const menu = screen.getByRole('menu')
    expect(menu.getAttribute('role')).toBe('menu')
    expect(screen.getAllByRole('menuitem').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByTestId('item-a').getAttribute('role')).toBe('menuitem')
  })

  it('reflects open state via aria-expanded on the trigger', () => {
    renderMenu()
    const trigger = screen.getByTestId('trigger')
    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
  })

  it('item click fires onClick and closes the menu', () => {
    const onSelect = vi.fn()
    render(
      <DropdownMenu>
        <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem data-testid="item" onClick={onSelect}>
            Item A
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    fireEvent.click(screen.getByTestId('item'))
    expect(onSelect).toHaveBeenCalled()
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('closes on Escape', () => {
    const onOpenChange = vi.fn()
    renderMenu({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('menu')).toBeTruthy()

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('menu')).toBeNull()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('closes when clicking outside the menu', () => {
    renderMenu()
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('menu')).toBeTruthy()

    fireEvent.mouseDown(document.body)
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('fires onOpenChange when toggling', () => {
    const onOpenChange = vi.fn()
    renderMenu({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('disabled item does not fire onClick and stays open', () => {
    const onSelect = vi.fn()
    render(
      <DropdownMenu>
        <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem data-testid="item" disabled onClick={onSelect}>
            Disabled item
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    const item = screen.getByTestId('item')
    expect(item.getAttribute('aria-disabled')).toBe('true')
    expect(item.getAttribute('data-disabled')).toBe('')
    fireEvent.click(item)
    expect(onSelect).not.toHaveBeenCalled()
    expect(screen.getByRole('menu')).toBeTruthy()
  })

  it('checkbox item fires onCheckedChange with the next value', () => {
    const onCheckedChange = vi.fn()
    render(
      <DropdownMenu>
        <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuCheckboxItem checked={false} onCheckedChange={onCheckedChange} data-testid="toggle">
            Toggle option
          </DropdownMenuCheckboxItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    fireEvent.click(screen.getByTestId('toggle'))
    expect(onCheckedChange).toHaveBeenCalledWith(true)
  })

  it('radio item fires onValueChange with its value', () => {
    const onValueChange = vi.fn()
    render(
      <DropdownMenu>
        <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuRadioGroup value="a" onValueChange={onValueChange}>
            <DropdownMenuRadioItem value="a" data-testid="radio-a">
              Alpha
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="b" data-testid="radio-b">
              Beta
            </DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByTestId('radio-a').getAttribute('role')).toBe('menuitemradio')
    expect(screen.getByTestId('radio-a').getAttribute('aria-checked')).toBe('true')
    expect(screen.getByTestId('radio-b').getAttribute('aria-checked')).toBe('false')
    fireEvent.click(screen.getByTestId('radio-b'))
    expect(onValueChange).toHaveBeenCalledWith('b')
  })

  it('radio item renders its children label', () => {
    render(
      <DropdownMenu>
        <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuRadioGroup value="a">
            <DropdownMenuRadioItem value="a">Alpha</DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="b">Beta</DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByText('Alpha')).toBeTruthy()
    expect(screen.getByText('Beta')).toBeTruthy()
    expect(screen.getByText('Alpha').closest('[role="menuitemradio"]')).toBeTruthy()
  })

  it('fires onSelect when an item is clicked', () => {
    const onSelect = vi.fn()
    render(
      <DropdownMenu>
        <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem data-testid="item" onSelect={onSelect}>
            Item A
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    fireEvent.click(screen.getByTestId('item'))
    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('re-measures after anchoring so the menu aligns to the trigger, not the viewport floor', () => {
    // jsdom has no layout, so fake the one measurement that matters. The menu
    // reports body width on its first read — it is not `position: fixed` yet,
    // so it is still a full-width block — and its real shrink-to-fit width on
    // every read after the anchor has been applied.
    let reads = 0
    const spy = vi
      .spyOn(window.HTMLElement.prototype, 'offsetWidth', 'get')
      .mockImplementation(function (this: HTMLElement) {
        if (this.getAttribute('role') !== 'menu') return 0
        return reads++ === 0 ? 800 : 202
      })

    try {
      renderMenu()
      const trigger = screen.getByTestId('trigger')
      // jsdom rects are all zero; put the trigger where the real one sits.
      trigger.getBoundingClientRect = () =>
        ({
          bottom: 247,
          height: 24,
          left: 208,
          right: 232,
          top: 223,
          width: 24,
          x: 208,
          y: 223,
          toJSON: () => ({}),
        }) as DOMRect

      fireEvent.click(trigger)
      const menu = screen.getByRole('menu')

      // 232 - 800 clamps to the 8px viewport floor; the stale read is corrected
      // by the post-commit re-measure, leaving `trigger.right - shrinkToFit`.
      expect(menu.style.left).toBe(`${232 - 202}px`)
      expect(menu.style.top).toBe('253px')
      expect(menu.style.position).toBe('fixed')
      expect(reads).toBeGreaterThan(1)
    } finally {
      spy.mockRestore()
    }
  })

  it('respects controlled open=false', () => {
    const onOpenChange = vi.fn()
    render(
      <DropdownMenu open={false} onOpenChange={onOpenChange}>
        <DropdownMenuTrigger data-testid="trigger">Open menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Item A</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
    expect(screen.queryByRole('menu')).toBeNull()
  })
})
