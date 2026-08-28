'use client'

import * as React from 'react'
import { cn } from '../../lib/cn'

export interface CheckboxProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type' | 'size'> {
  onCheckedChange?: (checked: boolean) => void
  size?: 'sm' | 'default' | 'lg'
  label?: React.ReactNode
  description?: string
  indeterminate?: boolean
}

const sizeMap = {
  sm: { box: 'h-4 w-4', icon: 12, stroke: 2 },
  default: { box: 'h-[18px] w-[18px]', icon: 14, stroke: 2 },
  lg: { box: 'h-5 w-5', icon: 16, stroke: 2.5 },
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

    React.useEffect(() => {
      const el = innerRef.current
      if (!el) return
      el.indeterminate = indeterminate ?? false
    }, [indeterminate])

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange?.(e)
      const next = e.target.checked
      if (!isControlled) setInternalChecked(next)
      onCheckedChange?.(next)
    }

    const s = sizeMap[size]
    const showCheck = isChecked && !indeterminate

    const input = (
      <span className={cn('relative inline-flex shrink-0 items-center justify-center', s.box, disabled && 'opacity-50')}>
        <input
          ref={(node) => {
            ;(innerRef as React.MutableRefObject<HTMLInputElement | null>).current = node
            if (typeof ref === 'function') ref(node)
            else if (ref) ref.current = node
          }}
          type="checkbox"
          id={id}
          disabled={disabled}
          checked={isChecked}
          className={cn(
            'peer absolute inset-0 cursor-pointer appearance-none',
            'rounded border transition-all duration-150',
            isChecked
              ? 'border-primary bg-primary'
              : 'border-input bg-background',
            'hover:border-primary/60',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
            'disabled:cursor-not-allowed disabled:hover:border-input',
            !label && className,
          )}
          onChange={handleChange}
          {...props}
        />
        {/* Checkmark */}
        <svg
          className={cn(
            'pointer-events-none absolute text-primary-foreground',
            showCheck ? 'scale-100 opacity-100' : 'scale-50 opacity-0',
            'transition-all duration-150',
          )}
          width={s.icon}
          height={s.icon}
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M3.5 8.5L6.5 11.5L12.5 4.5"
            stroke="currentColor"
            strokeWidth={s.stroke}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        {/* Indeterminate dash */}
        {indeterminate && (
          <svg
            className="pointer-events-none absolute text-primary-foreground"
            width={s.icon}
            height={s.icon}
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M4 8H12"
              stroke="currentColor"
              strokeWidth={s.stroke}
              strokeLinecap="round"
            />
          </svg>
        )}
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
Checkbox.displayName = 'Checkbox'

export { Checkbox }
