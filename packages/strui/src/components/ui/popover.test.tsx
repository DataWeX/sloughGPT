import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { Popover, PopoverTrigger, PopoverContent, PopoverClose } from './popover'

function stubLayout(rect: Partial<DOMRect>) {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0, x: 0, y: 0, toJSON: () => ({}),
    ...rect,
  } as DOMRect)
}

function renderPopover(props: { onOpenChange?: (open: boolean) => void } = {}) {
  return render(
    <div>
      <div data-testid="outside" />
      <Popover {...props}>
        <PopoverTrigger data-testid="trigger">Open popover</PopoverTrigger>
        <PopoverContent>
          <span>Popover content</span>
        </PopoverContent>
      </Popover>
    </div>,
  )
}

describe('Popover', () => {
  beforeEach(() => {
    stubLayout({})
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders a trigger with popover semantics', () => {
    renderPopover()
    const trigger = screen.getByTestId('trigger')
    expect(trigger.getAttribute('type')).toBe('button')
    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog')
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
  })

  it('does not render content while closed', () => {
    renderPopover()
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.queryByText('Popover content')).toBeNull()
  })

  it('opens on trigger click and renders content', () => {
    renderPopover()
    fireEvent.click(screen.getByTestId('trigger'))
    const content = screen.getByRole('dialog')
    expect(content.getAttribute('role')).toBe('dialog')
    expect(content.getAttribute('aria-modal')).toBe('false')
    expect(content.getAttribute('data-state')).toBe('open')
    expect(screen.getByText('Popover content')).toBeTruthy()
  })

  it('reflects open state via aria-expanded on the trigger', () => {
    renderPopover()
    const trigger = screen.getByTestId('trigger')
    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
  })

  it('positions content below the trigger', () => {
    stubLayout({ top: 100, bottom: 140, left: 30, width: 200, height: 40 })
    renderPopover()
    fireEvent.click(screen.getByTestId('trigger'))
    const content = screen.getByRole('dialog')
    expect(content.style.position).toBe('absolute')
    expect(content.style.top).toBe('148px')
    expect(content.style.left).toBe('30px')
  })

  it('closes on Escape', () => {
    const onOpenChange = vi.fn()
    renderPopover({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('dialog')).toBeTruthy()

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('closes when clicking outside', () => {
    const onOpenChange = vi.fn()
    renderPopover({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('dialog')).toBeTruthy()

    fireEvent.mouseDown(screen.getByTestId('outside'))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('PopoverClose closes the popover', () => {
    render(
      <div>
        <Popover>
          <PopoverTrigger data-testid="trigger">Open popover</PopoverTrigger>
          <PopoverContent>
            <span>Popover content</span>
            <PopoverClose data-testid="close">Close</PopoverClose>
          </PopoverContent>
        </Popover>
      </div>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('dialog')).toBeTruthy()
    fireEvent.click(screen.getByTestId('close'))
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('fires onOpenChange when opening and closing', () => {
    const onOpenChange = vi.fn()
    renderPopover({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('respects controlled open=false and does not render content', () => {
    const onOpenChange = vi.fn()
    render(
      <div>
        <Popover open={false} onOpenChange={onOpenChange}>
          <PopoverTrigger data-testid="trigger">Open popover</PopoverTrigger>
          <PopoverContent>
            <span>Popover content</span>
          </PopoverContent>
        </Popover>
      </div>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('renders content when controlled open is true', () => {
    render(
      <div>
        <Popover open={true}>
          <PopoverTrigger data-testid="trigger">Open popover</PopoverTrigger>
          <PopoverContent>
            <span>Popover content</span>
          </PopoverContent>
        </Popover>
      </div>,
    )
    expect(screen.getByRole('dialog')).toBeTruthy()
  })
})
