'use client'

import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'

export const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98]',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground shadow-sm hover:opacity-90 hover:shadow-md',
        secondary:
          'border border-border bg-secondary text-secondary-foreground shadow-sm hover:bg-primary/8 hover:text-primary hover:border-primary/25',
        ghost: 'text-muted-foreground hover:text-foreground hover:bg-accent/8',
        outline:
          'border border-border bg-transparent text-foreground shadow-sm hover:bg-primary/8 hover:text-primary hover:border-primary/25',
        destructive: 'bg-destructive text-destructive-foreground shadow-sm hover:opacity-90',
        menu: 'text-foreground hover:bg-primary/8 hover:text-primary focus-visible:ring-primary/40',
        bare: 'text-foreground hover:text-primary focus-visible:ring-primary/40',
        select: 'text-foreground hover:bg-primary/8 focus-visible:ring-primary/40 border border-border/50 px-2 py-1 rounded-lg',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 px-3 text-xs',
        lg: 'h-11 px-6',
        icon: 'h-10 w-11',
        'icon-sm': 'h-7 w-7',
        'icon-lg': 'h-11 w-12',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, type, ...props }, ref) => (
    <button
      ref={ref}
      type={type ?? 'button'}
      className={[buttonVariants({ variant, size }), className].filter(Boolean).join(' ')}
      {...props}
    />
  )
)
Button.displayName = 'Button'
