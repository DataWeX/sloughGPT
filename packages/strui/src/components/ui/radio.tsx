'use client'

import * as React from 'react'
import { cn } from '../../lib/cn'

export interface RadioProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type' | 'size'> {
  onCheckedChange?: (checked: boolean) => void
  size?: 'sm' | 'default' | 'lg'
  label?: React.ReactNode
  description?: string
}

const sizeMap = {
  sm: { box: 'h-4 w-4', dot: 6, ring: 7 },
  default: { box: 'h-[18px] w-[18px]', dot: 7, ring: 8 },
  lg: { box: 'h-5 w-5', dot: 8, ring: 9 },
}

const Radio = React.forwardRef<HTMLInputElement, RadioProps>(
  (
    {
      className,
      onCheckedChange,
      onChange,
      size = 'default',
      label,
      description,
      id,
      disabled,
      checked,
      defaultChecked,
      ...props
    },
    ref,
  ) => {
    const innerRef = React.useRef<HTMLInputElement | null>(null)
    const [internalChecked, setInternalChecked] = React.useState(defaultChecked ?? false)
    const isControlled = checked !== undefined
    const isChecked = isControlled ? checked : internalChecked

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange?.(e)
      const next = e.target.checked
      if (!isControlled) setInternalChecked(next)
      onCheckedChange?.(next)
    }

    const s = sizeMap[size]

    const input = (
      <span className={cn('relative inline-flex shrink-0 items-center justify-center', s.box, disabled && 'opacity-50')}>
        <input
          ref={(node) => {
            ;(innerRef as React.MutableRefObject<HTMLInputElement | null>).current = node
            if (typeof ref === 'function') ref(node)
            else if (ref) ref.current = node
          }}
          type="radio"
          id={id}
          disabled={disabled}
          checked={isChecked}
          className={cn(
            'peer absolute inset-0 cursor-pointer appearance-none',
            'rounded-full border transition-all duration-150',
            isChecked
              ? 'border-primary bg-background'
              : 'border-input bg-background',
            'hover:border-primary/60',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
            'disabled:cursor-not-allowed disabled:hover:border-input',
            !label && className,
          )}
          onChange={handleChange}
          {...props}
        />
        {/* Outer ring */}
        <svg
          className="pointer-events-none absolute"
          width={s.box}
          height={s.box}
          viewBox="0 0 18 18"
          fill="none"
          aria-hidden="true"
        >
          <circle
            cx="9"
            cy="9"
            r="8"
            className={cn(
              'transition-all duration-150',
              isChecked ? 'stroke-primary' : 'stroke-transparent',
            )}
            strokeWidth="1"
          />
        </svg>
        {/* Inner dot */}
        <svg
          className={cn(
            'pointer-events-none absolute text-primary',
            isChecked ? 'scale-100 opacity-100' : 'scale-0 opacity-0',
            'transition-all duration-150',
          )}
          width={s.dot}
          height={s.dot}
          viewBox="0 0 8 8"
          fill="none"
          aria-hidden="true"
        >
          <circle cx="4" cy="4" r="4" fill="currentColor" />
        </svg>
      </span>
    )

    if (!label && !description) return input

    return (
      <label
        htmlFor={id}
        className={cn(
          'inline-flex items-start gap-2.5',
          disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
          className,
        )}
      >
        <span className="mt-0.5">{input}</span>
        <span className="flex flex-col gap-0.5">
          {label && (
            <span className="text-sm font-medium leading-snug text-foreground select-none">
              {label}
            </span>
          )}
          {description && (
            <span className="text-xs text-muted-foreground leading-relaxed">
              {description}
            </span>
          )}
        </span>
      </label>
    )
  },
)
Radio.displayName = 'Radio'

export { Radio }
