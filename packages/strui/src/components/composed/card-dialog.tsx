'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'

export interface CardDialogProps {
  /** Whether the dialog is open */
  open: boolean
  /** Callback to toggle open state */
  onOpenChange: (open: boolean) => void
  /** Dialog title */
  title: string
  /** Optional description below the title */
  description?: string
  /** Dialog body content */
  children: ReactNode
  /** Optional footer content (buttons, actions) */
  footer?: ReactNode
  /** Additional CSS classes for DialogContent */
  className?: string
  /** Maximum width variant */
  size?: 'sm' | 'md' | 'lg' | 'xl'
  /** Test ID for testing */
  testId?: string
}

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
}

/**
 * Dialog with header, content, and footer sections.
 *
 * Thin wrapper around Dialog that provides a consistent layout
 * for modal dialogs with title, description, body, and footer.
 *
 * @example
 * ```tsx
 * <CardDialog
 *   open={showDialog}
 *   onOpenChange={setShowDialog}
 *   title="Import Dataset"
 *   description="Choose a data source to import"
 *   footer={
 *     <>
 *       <Button variant="outline" onClick={onCancel}>Cancel</Button>
 *       <Button onClick={onImport}>Import</Button>
 *     </>
 *   }
 * >
 *   <DatasetPicker onSelect={setSelected} />
 * </CardDialog>
 * ```
 */
export function CardDialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  className,
  size = 'md',
  testId,
}: CardDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className={cn(sizeClasses[size], className)} data-testid={testId}>
        <DialogHeader>
          <DialogTitle className="text-base">{title}</DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>
        {children}
        {footer && <DialogFooter>{footer}</DialogFooter>}
      </DialogContent>
    </Dialog>
  )
}
