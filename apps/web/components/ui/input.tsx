'use client'

import { forwardRef, type InputHTMLAttributes, type ChangeEventHandler } from 'react'

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

export type InputProps = InputHTMLAttributes<HTMLInputElement>

const Input = forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={[inputFieldClassName, 'h-10', className].filter(Boolean).join(' ')}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = 'Input'

export interface SearchInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  iconClass?: string
  value?: string
  onChange?: (value: string) => void
}

export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  function SearchInput({ className, iconClass, value, onChange, ...props }, ref) {
    const handleChange: ChangeEventHandler<HTMLInputElement> = (e) => {
      onChange?.(e.target.value)
    }
    return (
      <div className="relative">
        <svg
          className={[
            "absolute left-2 top-1/2 -translate-y-1/2 h-2.5 w-2.5 text-muted-foreground pointer-events-none",
            iconClass,
          ].filter(Boolean).join(' ')}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          ref={ref}
          type="text"
          value={value}
          onChange={handleChange}
          className={[inputFieldClassName, "pl-6 pr-2 py-1 text-xs", className].filter(Boolean).join(' ')}
          {...props}
        />
      </div>
    )
  }
)

export { Input }
