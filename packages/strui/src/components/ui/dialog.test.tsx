import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import {
  Dialog,
  DialogTrigger,
  DialogOverlay,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from './dialog'

function renderDialog(props: { onOpenChange?: (open: boolean) => void } = {}) {
  return render(
    <Dialog {...props}>
      <DialogTrigger data-testid="trigger">Open dialog</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Dialog title</DialogTitle>
          <DialogDescription>Dialog description</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose>Cancel</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>,
  )
}

describe('Dialog', () => {
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

  it('renders a trigger with dialog semantics', () => {
    renderDialog()
    const trigger = screen.getByTestId('trigger')
    expect(trigger.getAttribute('type')).toBe('button')
    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog')
  })

  it('does not render content while closed', () => {
    renderDialog()
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('opens on trigger click with role dialog and aria-modal', () => {
    renderDialog()
    fireEvent.click(screen.getByTestId('trigger'))
    const dialog = screen.getByRole('dialog')
    expect(dialog.getAttribute('role')).toBe('dialog')
    expect(dialog.getAttribute('aria-modal')).toBe('true')
  })

  it('links title and description via aria-labelledby and aria-describedby', () => {
    renderDialog()
    fireEvent.click(screen.getByTestId('trigger'))
    const dialog = screen.getByRole('dialog')
    expect(dialog.getAttribute('aria-labelledby')).toBeTruthy()
    expect(dialog.getAttribute('aria-describedby')).toBeTruthy()
    expect(screen.getByText('Dialog title').id).toBe(dialog.getAttribute('aria-labelledby'))
    expect(screen.getByText('Dialog description').id).toBe(dialog.getAttribute('aria-describedby'))
  })

  it('closes on Escape', () => {
    const onOpenChange = vi.fn()
    renderDialog({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('dialog')).toBeTruthy()

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('renders a close button with aria-label when open', () => {
    renderDialog()
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('button', { name: 'Close' })).toBeTruthy()
  })

  it('closes via DialogClose button', () => {
    const onOpenChange = vi.fn()
    renderDialog({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('fires onOpenChange when opening', () => {
    const onOpenChange = vi.fn()
    renderDialog({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
  })

  it('respects controlled open=false and does not render content', () => {
    const onOpenChange = vi.fn()
    render(
      <Dialog open={false} onOpenChange={onOpenChange}>
        <DialogTrigger data-testid="trigger">Open dialog</DialogTrigger>
        <DialogContent>Content</DialogContent>
      </Dialog>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('renders content when controlled open is true', () => {
    render(
      <Dialog open={true}>
        <DialogTrigger data-testid="trigger">Open dialog</DialogTrigger>
        <DialogContent>Content</DialogContent>
      </Dialog>,
    )
    expect(screen.getByRole('dialog')).toBeTruthy()
  })

  it('closes when the overlay is clicked', () => {
    const onOpenChange = vi.fn()
    render(
      <Dialog onOpenChange={onOpenChange}>
        <DialogTrigger data-testid="trigger">Open dialog</DialogTrigger>
        <DialogOverlay data-testid="overlay" />
        <DialogContent>Content</DialogContent>
      </Dialog>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('dialog')).toBeTruthy()
    fireEvent.click(screen.getByTestId('overlay'))
    expect(onOpenChange).toHaveBeenCalledWith(false)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('does not leak asChild to the DOM and forwards it to the child', () => {
    render(
      <Dialog>
        <DialogTrigger asChild data-testid="child">
          <button data-testid="custom-trigger">Custom</button>
        </DialogTrigger>
        <DialogContent>Content</DialogContent>
      </Dialog>,
    )
    const custom = screen.getByTestId('custom-trigger')
    expect(custom).toBeTruthy()
    expect(screen.queryByTestId('child')).toBeNull()
    expect(custom.hasAttribute('asChild')).toBe(false)
    fireEvent.click(custom)
    expect(screen.getByRole('dialog')).toBeTruthy()
  })
})
