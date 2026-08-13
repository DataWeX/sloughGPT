import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi, afterEach } from 'vitest'

import { SimpleTooltip, Tooltip, TooltipContent, TooltipTrigger } from './tooltip'

function stubLayout() {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    top: 0, bottom: 0, left: 0, right: 100, width: 100, height: 40, x: 0, y: 0, toJSON: () => ({}),
  } as DOMRect)
}

function queryContent(): HTMLElement | null {
  return document.body.querySelector<HTMLElement>('[data-testid="content"]')
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('Tooltip', () => {
  it('renders the trigger as a button', () => {
    const { container } = render(
      <Tooltip>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const trigger = screen.getByTestId('trigger')
    expect(trigger.tagName).toBe('BUTTON')
    expect(trigger.getAttribute('type')).toBe('button')
    expect(trigger.textContent).toBe('Hover')
    expect(container.querySelector('[role="tooltip"]')).toBeNull()
  })

  it('does not leak the asChild prop to the DOM', () => {
    render(
      <Tooltip>
        <TooltipTrigger asChild data-testid="trigger">
          Hover
        </TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const trigger = screen.getByTestId('trigger')
    expect(trigger.hasAttribute('aschild')).toBe(false)
    expect(trigger.hasAttribute('asChild')).toBe(false)
  })

  it('recomputes position when opened from the start', () => {
    stubLayout()
    render(
      <Tooltip defaultOpen>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent side="bottom" data-testid="content">
          Tip text
        </TooltipContent>
      </Tooltip>,
    )
    const content = queryContent()!
    expect(content.style.position).toBe('absolute')
    expect(content.style.top).toBe('8px')
    expect(content.style.left).toBe('8px')
  })

  it('does not render content by default', () => {
    render(
      <Tooltip>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    expect(document.body.querySelector('[role="tooltip"]')).toBeNull()
  })

  it('renders content when defaultOpen is true', () => {
    stubLayout()
    render(
      <Tooltip defaultOpen>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const content = queryContent()!
    expect(content).toBeTruthy()
    expect(content.textContent).toContain('Tip text')
  })

  it('renders content when controlled open', () => {
    stubLayout()
    render(
      <Tooltip open>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    expect(document.body.querySelector('[role="tooltip"]')).toBeTruthy()
  })

  it('links the trigger to the content via aria-describedby', () => {
    stubLayout()
    render(
      <Tooltip defaultOpen>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const trigger = screen.getByTestId('trigger')
    const content = queryContent()!
    expect(trigger.getAttribute('aria-describedby')).toBe(content.id)
  })

  it('marks content with tooltip role and open state', () => {
    stubLayout()
    render(
      <Tooltip defaultOpen>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const content = queryContent()!
    expect(content.getAttribute('role')).toBe('tooltip')
    expect(content.getAttribute('data-state')).toBe('open')
  })

  it('opens on focus and closes on blur', () => {
    stubLayout()
    const onOpenChange = vi.fn()
    render(
      <Tooltip onOpenChange={onOpenChange}>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const trigger = screen.getByTestId('trigger')
    fireEvent.focus(trigger)
    expect(onOpenChange).toHaveBeenCalledWith(true)
    expect(document.body.querySelector('[role="tooltip"]')).toBeTruthy()
    fireEvent.blur(trigger)
    expect(onOpenChange).toHaveBeenCalledWith(false)
    expect(document.body.querySelector('[role="tooltip"]')).toBeNull()
  })

  it('opens on hover after the delay', () => {
    vi.useFakeTimers()
    stubLayout()
    render(
      <Tooltip delayDuration={200}>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    fireEvent.mouseOver(screen.getByTestId('trigger'))
    expect(queryContent()).toBeNull()
    act(() => {
      vi.advanceTimersByTime(200)
    })
    expect(queryContent()).toBeTruthy()
  })

  it('closes on mouse leave', () => {
    vi.useFakeTimers()
    stubLayout()
    render(
      <Tooltip delayDuration={200}>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const trigger = screen.getByTestId('trigger')
    fireEvent.mouseOver(trigger)
    act(() => {
      vi.advanceTimersByTime(200)
    })
    expect(queryContent()).toBeTruthy()
    fireEvent.mouseOut(trigger)
    expect(queryContent()).toBeNull()
  })

  it('cancels the pending open timer on early mouse leave', () => {
    vi.useFakeTimers()
    stubLayout()
    render(
      <Tooltip delayDuration={200}>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    const trigger = screen.getByTestId('trigger')
    fireEvent.mouseOver(trigger)
    fireEvent.mouseOut(trigger)
    act(() => {
      vi.advanceTimersByTime(500)
    })
    expect(queryContent()).toBeNull()
  })

  it('positions content absolutely in a portal', () => {
    stubLayout()
    render(
      <Tooltip>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent data-testid="content">Tip text</TooltipContent>
      </Tooltip>,
    )
    fireEvent.focus(screen.getByTestId('trigger'))
    const content = queryContent()!
    expect(content.style.position).toBe('absolute')
    expect(content.style.top).toBe('8px')
    expect(content.style.left).toBe('8px')
  })

  it('positions content to the right side', () => {
    stubLayout()
    render(
      <Tooltip>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent side="right" data-testid="content">
          Tip text
        </TooltipContent>
      </Tooltip>,
    )
    fireEvent.focus(screen.getByTestId('trigger'))
    const content = queryContent()!
    expect(content.style.left).toBe('108px')
  })

  it('merges custom className on content', () => {
    stubLayout()
    render(
      <Tooltip defaultOpen>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent className="my-tip" data-testid="content">
          Tip text
        </TooltipContent>
      </Tooltip>,
    )
    const content = queryContent()!
    expect(content.classList.contains('my-tip')).toBe(true)
  })

  it('renders a muted variant on content', () => {
    stubLayout()
    render(
      <Tooltip defaultOpen>
        <TooltipTrigger data-testid="trigger">Hover</TooltipTrigger>
        <TooltipContent variant="muted" data-testid="content">
          Tip text
        </TooltipContent>
      </Tooltip>,
    )
    const content = queryContent()!
    expect(content.classList.contains('bg-card')).toBe(true)
  })
})

describe('SimpleTooltip', () => {
  it('wraps the child in a trigger button', () => {
    render(
      <SimpleTooltip content="Copy">
        <span>Button</span>
      </SimpleTooltip>,
    )
    const trigger = screen.getByText('Button').closest('button')
    expect(trigger).toBeTruthy()
  })

  it('shows content on hover', () => {
    vi.useFakeTimers()
    stubLayout()
    render(
      <SimpleTooltip content="Copy" delay={100}>
        <span>Button</span>
      </SimpleTooltip>,
    )
    fireEvent.mouseOver(screen.getByText('Button'))
    act(() => {
      vi.advanceTimersByTime(100)
    })
    const content = document.body.querySelector('[role="tooltip"]')!
    expect(content).toBeTruthy()
    expect(content.textContent).toContain('Copy')
  })
})
