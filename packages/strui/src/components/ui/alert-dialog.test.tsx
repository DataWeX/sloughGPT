import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import {
  AlertDialog,
  AlertDialogTrigger,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogAction,
  AlertDialogCancel,
} from './alert-dialog'

function renderAlertDialog(props: { onOpenChange?: (open: boolean) => void } = {}) {
  return render(
    <AlertDialog {...props}>
      <AlertDialogTrigger data-testid="trigger">Delete item</AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Are you sure?</AlertDialogTitle>
          <AlertDialogDescription>This action cannot be undone.</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction>Confirm</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>,
  )
}

describe('AlertDialog', () => {
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
    renderAlertDialog()
    const trigger = screen.getByTestId('trigger')
    expect(trigger.getAttribute('type')).toBe('button')
    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog')
  })

  it('does not render content while closed', () => {
    renderAlertDialog()
    expect(screen.queryByRole('alertdialog')).toBeNull()
  })

  it('opens on trigger click with role alertdialog and aria-modal', () => {
    renderAlertDialog()
    fireEvent.click(screen.getByTestId('trigger'))
    const dialog = screen.getByRole('alertdialog')
    expect(dialog.getAttribute('role')).toBe('alertdialog')
    expect(dialog.getAttribute('aria-modal')).toBe('true')
  })

  it('links title and description via aria-labelledby and aria-describedby', () => {
    renderAlertDialog()
    fireEvent.click(screen.getByTestId('trigger'))
    const dialog = screen.getByRole('alertdialog')
    expect(dialog.getAttribute('aria-labelledby')).toBeTruthy()
    expect(dialog.getAttribute('aria-describedby')).toBeTruthy()
    expect(screen.getByText('Are you sure?').id).toBe(dialog.getAttribute('aria-labelledby'))
    expect(screen.getByText('This action cannot be undone.').id).toBe(dialog.getAttribute('aria-describedby'))
  })

  it('closes on Escape', () => {
    const onOpenChange = vi.fn()
    renderAlertDialog({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(screen.getByRole('alertdialog')).toBeTruthy()

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('fires onOpenChange when opening', () => {
    const onOpenChange = vi.fn()
    renderAlertDialog({ onOpenChange })
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
  })

  it('cancel fires its onClick then closes', () => {
    const onOpenChange = vi.fn()
    const onCancel = vi.fn()
    render(
      <AlertDialog onOpenChange={onOpenChange}>
        <AlertDialogTrigger data-testid="trigger">Delete item</AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogCancel onClick={onCancel}>Cancel</AlertDialogCancel>
          <AlertDialogAction>Confirm</AlertDialogAction>
        </AlertDialogContent>
      </AlertDialog>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalled()
    expect(onOpenChange).toHaveBeenLastCalledWith(false)
    expect(screen.queryByRole('alertdialog')).toBeNull()
  })

  it('confirm fires its onClick then closes', () => {
    const onOpenChange = vi.fn()
    const onConfirm = vi.fn()
    render(
      <AlertDialog onOpenChange={onOpenChange}>
        <AlertDialogTrigger data-testid="trigger">Delete item</AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>Confirm</AlertDialogAction>
        </AlertDialogContent>
      </AlertDialog>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }))
    expect(onConfirm).toHaveBeenCalled()
    expect(onOpenChange).toHaveBeenLastCalledWith(false)
    expect(screen.queryByRole('alertdialog')).toBeNull()
  })

  it('respects controlled open=false and does not render content', () => {
    const onOpenChange = vi.fn()
    render(
      <AlertDialog open={false} onOpenChange={onOpenChange}>
        <AlertDialogTrigger data-testid="trigger">Delete item</AlertDialogTrigger>
        <AlertDialogContent>Content</AlertDialogContent>
      </AlertDialog>,
    )
    fireEvent.click(screen.getByTestId('trigger'))
    expect(onOpenChange).toHaveBeenCalledWith(true)
    expect(screen.queryByRole('alertdialog')).toBeNull()
  })

  it('renders content when controlled open is true', () => {
    render(
      <AlertDialog open={true}>
        <AlertDialogTrigger data-testid="trigger">Delete item</AlertDialogTrigger>
        <AlertDialogContent>Content</AlertDialogContent>
      </AlertDialog>,
    )
    expect(screen.getByRole('alertdialog')).toBeTruthy()
  })
})
