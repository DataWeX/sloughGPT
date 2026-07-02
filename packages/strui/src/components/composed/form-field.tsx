import type { ReactNode } from 'react'

import { Label } from '../ui/label'
import { cn } from '../../lib/cn'

export interface FormFieldProps {
  id: string
  label: string
  /** Shown below the control when present. */
  error?: string
  /** Hint below the field (hidden when `error` is set). */
  hint?: string
  children: ReactNode
  className?: string
}

/** Label + control + validation / hint stack for settings and AI tool forms. */
export function FormField({ id, label, error, hint, children, className }: FormFieldProps) {
  return (
    <div className={cn('space-y-1.5', className)}>
      <Label htmlFor={id}>{label}</Label>
      {children}
      {error ? (
        <p id={`${id}-error`} className="text-xs text-destructive" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-xs text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  )
}
