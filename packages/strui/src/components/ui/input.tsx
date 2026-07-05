import * as React from 'react'

import { cn } from '../../lib/cn'

export const inputFieldClassName = cn(
  'flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground shadow-sm',
  'transition-[border-color,box-shadow,background-color] duration-200',
  'placeholder:text-muted-foreground',
  'selection:bg-primary/20 selection:text-foreground',
  'hover:border-primary/50',
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
  'focus-visible:border-primary/60',
  'disabled:cursor-not-allowed disabled:opacity-50'
)

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>

const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(inputFieldClassName, 'h-10', className)}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = 'Input'

export { Input }
