'use client'

import { forwardRef, type LabelHTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

export interface LabelProps extends LabelHTMLAttributes<HTMLLabelElement> {
  /** Marks field as required with a red asterisk */
  required?: boolean
  /** Muted variant for secondary/helper labels */
  variant?: 'default' | 'muted' | 'uppercase'
}

const Label = forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, required, variant = 'default', children, ...props }, ref) => (
    <label
      ref={ref}
      className={cn(
        'inline-flex items-center gap-1 leading-none',
        'peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
        variant === 'default' && 'text-sm font-medium text-foreground',
        variant === 'muted' && 'text-xs font-medium text-muted-foreground',
        variant === 'uppercase' && 'text-[10px] font-semibold uppercase tracking-wider text-muted-foreground',
        className,
      )}
      {...props}
    >
      {children}
      {required && (
        <span className="text-destructive" aria-hidden="true">
          *
        </span>
      )}
    </label>
  ),
)
Label.displayName = 'Label'

export { Label }
