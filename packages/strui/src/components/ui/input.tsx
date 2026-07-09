'use client'

import { forwardRef, type InputHTMLAttributes, type ChangeEventHandler, type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export const inputFieldClassName = [
  'flex w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground shadow-sm',
  'transition-[border-color,box-shadow,background-color] duration-200',
  'placeholder:text-muted-foreground',
  'selection:bg-primary/20 selection:text-foreground',
  'hover:border-primary/50',
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
  'focus-visible:border-primary/60',
  'disabled:cursor-not-allowed disabled:opacity-50',
].join(' ')

/* ── Base Input ─────────────────────────────────────────────── */

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Error styling */
  error?: boolean
  /** Icon shown on the left */
  leftIcon?: ReactNode
  /** Icon or element shown on the right */
  rightElement?: ReactNode
}

const Input = forwardRef<HTMLInputElement, InputProps>(({ className, type, error, leftIcon, rightElement, ...props }, ref) => {
  if (leftIcon || rightElement) {
    return (
      <div className="relative flex items-center">
        {leftIcon && (
          <span className="absolute left-3 flex items-center text-muted-foreground pointer-events-none">
            {leftIcon}
          </span>
        )}
        <input
          type={type}
          className={cn(
            inputFieldClassName,
            'h-10',
            leftIcon && 'pl-9',
            rightElement && 'pr-9',
            error && 'border-destructive/60 focus-visible:ring-destructive/40 hover:border-destructive/70',
            className,
          )}
          ref={ref}
          {...props}
        />
        {rightElement && (
          <span className="absolute right-3 flex items-center text-muted-foreground">
            {rightElement}
          </span>
        )}
      </div>
    )
  }

  return (
    <input
      type={type}
      className={cn(
        inputFieldClassName,
        'h-10',
        error && 'border-destructive/60 focus-visible:ring-destructive/40 hover:border-destructive/70',
        className,
      )}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = 'Input'

/* ── Search Input ───────────────────────────────────────────── */

export interface SearchInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  iconClass?: string
  value?: string
  onChange?: (value: string) => void
  /** Show clear button when there's a value */
  clearable?: boolean
  onClear?: () => void
}

export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  function SearchInput({ className, iconClass, value, onChange, clearable = true, onClear, ...props }, ref) {
    const handleChange: ChangeEventHandler<HTMLInputElement> = (e) => {
      onChange?.(e.target.value)
    }
    const handleClear = () => {
      onChange?.('')
      onClear?.()
    }

    return (
      <div className="relative flex items-center">
        <svg
          className={cn(
            'absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none',
            iconClass,
          )}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          ref={ref}
          type="search"
          value={value}
          onChange={handleChange}
          className={cn(
            inputFieldClassName,
            'pl-8',
            clearable && value ? 'pr-7' : 'pr-2',
            'py-1.5 text-xs h-8',
            className,
          )}
          {...props}
        />
        {clearable && value && (
          <button
            type="button"
            aria-label="Clear search"
            onClick={handleClear}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors focus:outline-none"
          >
            <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    )
  }
)

export { Input }
