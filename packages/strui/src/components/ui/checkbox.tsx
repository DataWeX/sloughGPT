'use client'

import * as React from 'react'
import { cn } from '../../lib/cn'

export interface CheckboxProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type' | 'size'> {
  onCheckedChange?: (checked: boolean) => void
  /** Size variant */
  size?: 'sm' | 'default' | 'lg'
  /** Label rendered inline */
  label?: React.ReactNode
  /** Description rendered below label */
  description?: string
  /** Indeterminate state */
  indeterminate?: boolean
}

const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  (
    {
      className,
      onCheckedChange,
      onChange,
      size = 'default',
      label,
      description,
      indeterminate,
      id,
      disabled,
      ...props
    },
    ref,
  ) => {
    const innerRef = React.useRef<HTMLInputElement | null>(null)

    // Sync indeterminate imperatively (no HTML attribute for it)
    React.useEffect(() => {
      const el = innerRef.current
      if (!el) return
      el.indeterminate = indeterminate ?? false
    }, [indeterminate])

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange?.(e)
      onCheckedChange?.(e.target.checked)
    }

    const sizeMap = {
      sm: 'h-3.5 w-3.5',
      default: 'h-4 w-4',
      lg: 'h-5 w-5',
    }

    const input = (
      <input
        ref={(node) => {
          ;(innerRef as React.MutableRefObject<HTMLInputElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        type="checkbox"
        id={id}
        disabled={disabled}
        className={cn(
          sizeMap[size],
          'shrink-0 rounded-sm border border-input',
          'bg-background text-primary',
          'transition-colors duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
          'checked:bg-primary checked:border-primary',
          'indeterminate:bg-primary/60 indeterminate:border-primary/60',
          'disabled:cursor-not-allowed disabled:opacity-50',
          // Custom check indicator via accent-color
          'accent-primary',
          !label && className,
        )}
        onChange={handleChange}
        {...props}
      />
    )

    if (!label && !description) return input

    return (
      <div className={cn('flex items-start gap-2.5', className)}>
        <div className="mt-0.5">{input}</div>
        <div className="flex flex-col gap-0.5">
          {label && (
            <label
              htmlFor={id}
              className={cn(
                'text-sm font-medium leading-snug text-foreground select-none',
                disabled && 'cursor-not-allowed opacity-50',
              )}
            >
              {label}
            </label>
          )}
          {description && (
            <p className={cn('text-xs text-muted-foreground leading-relaxed', disabled && 'opacity-50')}>
              {description}
            </p>
          )}
        </div>
      </div>
    )
  },
)
Checkbox.displayName = 'Checkbox'

export { Checkbox }
