'use client'

import { forwardRef, useCallback, type InputHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

interface SwitchProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  checked?: boolean
  defaultChecked?: boolean
  onCheckedChange?: (checked: boolean) => void
}

const Switch = forwardRef<HTMLInputElement, SwitchProps>(
  ({ className, checked: controlledChecked, defaultChecked = false, onCheckedChange, disabled, ...props }, ref) => {
    const handleClick = useCallback(() => {
      if (disabled) return
      const next = !controlledChecked
      onCheckedChange?.(next)
    }, [controlledChecked, disabled, onCheckedChange])

    return (
      <button
        type="button"
        role="switch"
        aria-checked={controlledChecked}
        aria-disabled={disabled}
        disabled={disabled}
        onClick={handleClick}
        className={cn(
          'peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
          'disabled:cursor-not-allowed disabled:opacity-50',
          controlledChecked ? 'bg-primary' : 'bg-muted',
          className,
        )}
      >
        <span
          className={cn(
            'pointer-events-none block h-5 w-5 rounded-full bg-white shadow-md ring-0 transition-transform duration-200',
            controlledChecked ? 'translate-x-5' : 'translate-x-0',
          )}
        />
        <input
          ref={ref}
          type="checkbox"
          className="sr-only"
          checked={controlledChecked}
          onChange={() => {}}
          tabIndex={-1}
          {...props}
        />
      </button>
    )
  },
)
Switch.displayName = 'Switch'

export { Switch }
