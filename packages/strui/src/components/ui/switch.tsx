'use client'

import { forwardRef, useCallback, useState, type InputHTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

export interface SwitchProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange' | 'size'> {
  checked?: boolean
  defaultChecked?: boolean
  onCheckedChange?: (checked: boolean) => void
  /** Smaller variant for dense layouts */
  size?: 'sm' | 'default'
  /** Optional label text to display inline */
  label?: string
}

const Switch = forwardRef<HTMLInputElement, SwitchProps>(
  (
    {
      className,
      checked: controlledChecked,
      defaultChecked = false,
      onCheckedChange,
      disabled,
      size = 'default',
      label,
      id,
      ...props
    },
    ref
  ) => {
    // Uncontrolled state when checked is not provided
    const [internalChecked, setInternalChecked] = useState(defaultChecked)
    const isControlled = controlledChecked !== undefined
    const isChecked = isControlled ? controlledChecked : internalChecked

    const handleClick = useCallback(() => {
      if (disabled) return
      const next = !isChecked
      if (!isControlled) setInternalChecked(next)
      onCheckedChange?.(next)
    }, [isChecked, isControlled, disabled, onCheckedChange])

    const trackSizes = size === 'sm' ? 'h-4 w-7' : 'h-6 w-11'
    const thumbSizes = size === 'sm' ? 'h-3 w-3' : 'h-5 w-5'
    const thumbTranslate = size === 'sm'
      ? isChecked ? 'translate-x-3' : 'translate-x-0'
      : isChecked ? 'translate-x-5' : 'translate-x-0'

    return (
      <div className={cn('inline-flex items-center gap-2', className)}>
        <button
          type="button"
          role="switch"
          aria-checked={isChecked}
          aria-disabled={disabled}
          disabled={disabled}
          onClick={handleClick}
          className={cn(
            'peer inline-flex shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200',
            'hover:brightness-110',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background',
            'disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:brightness-100',
            trackSizes,
            isChecked ? 'bg-primary' : 'bg-muted',
          )}
        >
          <span
            className={cn(
              'pointer-events-none block rounded-full bg-white shadow-md ring-0 transition-transform duration-200',
              'disabled:opacity-40',
              thumbSizes,
              thumbTranslate,
            )}
          />
          {/* Hidden input for form submission */}
          <input
            ref={ref}
            type="checkbox"
            id={id}
            className="sr-only"
            checked={isChecked}
            onChange={() => {}}
            tabIndex={-1}
            disabled={disabled}
            {...props}
          />
        </button>
        {label && (
          <label
            htmlFor={id}
            className={cn(
              'text-sm text-foreground select-none',
              disabled && 'cursor-not-allowed opacity-40',
            )}
          >
            {label}
          </label>
        )}
      </div>
    )
  }
)
Switch.displayName = 'Switch'

export { Switch }
