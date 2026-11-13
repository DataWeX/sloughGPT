import { describe, expect, it, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { Collapsible, CollapsibleTrigger, CollapsibleContent } from './collapsible'

function renderCollapsible(props: { onOpenChange?: (open: boolean) => void } = {}) {
  return render(
    <Collapsible {...props}>
      <CollapsibleTrigger data-testid="trigger">Toggle section</CollapsibleTrigger>
      <CollapsibleContent>Collapsible body</CollapsibleContent>
    </Collapsible>,
  )
}

describe('Collapsible', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders trigger and content', () => {
    renderCollapsible()
    expect(screen.getByTestId('trigger').getAttribute('type')).toBe('button')
    expect(screen.getByText('Collapsible body')).toBeTruthy()
  })

  it('starts closed by default', () => {
    renderCollapsible()
    expect(screen.getByTestId('trigger').getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByTestId('trigger').getAttribute('data-state')).toBe('closed')
    expect(screen.getByRole('region').getAttribute('data-state')).toBe('closed')
  })

  it('toggles open on trigger click', () => {
    renderCollapsible()
    const trigger = screen.getByTestId('trigger')
    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    expect(trigger.getAttribute('data-state')).toBe('open')
    expect(screen.getByRole('region').getAttribute('data-state')).toBe('open')
  })

  it('toggles closed on second click', () => {
    renderCollapsible()
    const trigger = screen.getByTestId('trigger')
    fireEvent.click(trigger)
    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByRole('region').getAttribute('data-state')).toBe('closed')
  })

  it('fires onOpenChange when toggling', () => {
    const onOpenChange = vi.fn()
    renderCollapsible({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('renders open when defaultOpen is true', () => {
    render(<Collapsible defaultOpen={true}>
      <CollapsibleTrigger data-testid="trigger">Toggle section</CollapsibleTrigger>
      <CollapsibleContent>Collapsible body</CollapsibleContent>
    </Collapsible>)
    expect(screen.getByTestId('trigger').getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByRole('region').getAttribute('data-state')).toBe('open')
  })

  it('respects controlled open', () => {
    const onOpenChange = vi.fn()
    render(
      <Collapsible open={true} onOpenChange={onOpenChange}>
        <CollapsibleTrigger data-testid="trigger">Toggle section</CollapsibleTrigger>
        <CollapsibleContent>Collapsible body</CollapsibleContent>
      </Collapsible>,
    )
    const trigger = screen.getByTestId('trigger')
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(trigger)
    expect(onOpenChange).toHaveBeenCalledWith(false)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
  })

  it('does not toggle when disabled', () => {
    const onOpenChange = vi.fn()
    render(
      <Collapsible disabled onOpenChange={onOpenChange}>
        <CollapsibleTrigger data-testid="trigger">Toggle section</CollapsibleTrigger>
        <CollapsibleContent>Collapsible body</CollapsibleContent>
      </Collapsible>,
    )
    const trigger = screen.getByTestId('trigger')
    expect(trigger.getAttribute('disabled')).toBe('')
    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(onOpenChange).not.toHaveBeenCalled()
  })
})
