'use client'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '../ui/alert-dialog'
import { cn } from '../../lib/cn'

export interface ConfirmDialogProps {
  /** Whether the dialog is open */
  open: boolean
  /** Callback to toggle open state */
  onOpenChange: (open: boolean) => void
  /** Dialog title */
  title: string
  /** Dialog description / body text */
  description: string
  /** Label for the confirm button (default: "Confirm") */
  confirmLabel?: string
  /** Label for the cancel button (default: "Cancel") */
  cancelLabel?: string
  /** Callback when confirm is clicked */
  onConfirm: () => void
  /** Whether the confirm button should be destructive styled (default: true) */
  destructive?: boolean
  /** Additional CSS classes for the confirm button */
  confirmClassName?: string
}

/**
 * Confirmation dialog with confirm/cancel actions.
 *
 * Thin wrapper around AlertDialog that provides a consistent
 * confirm/cancel pattern for destructive or important actions.
 *
 * @example
 * ```tsx
 * <ConfirmDialog
 *   open={showDelete}
 *   onOpenChange={setShowDelete}
 *   title="Delete checkpoint?"
 *   description="This action cannot be undone."
 *   confirmLabel="Delete"
 *   onConfirm={() => deleteCheckpoint(id)}
 * />
 * ```
 */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  destructive = true,
  confirmClassName,
}: ConfirmDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{cancelLabel}</AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            className={cn(
              destructive && 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
              confirmClassName
            )}
          >
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
