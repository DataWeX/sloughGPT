'use client'

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../lib/cn'

export const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 ease-smooth focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-40 active:scale-[0.98]',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground shadow-sm hover:bg-primary/90 hover:shadow-md',
        secondary:
          'border border-border bg-secondary text-secondary-foreground shadow-sm hover:bg-primary/8 hover:text-primary hover:border-primary/25',
        ghost: 'text-muted-foreground hover:text-foreground hover:bg-accent/8',
        outline:
          'border border-border bg-transparent text-foreground shadow-sm hover:bg-primary/8 hover:text-primary hover:border-primary/25',
        destructive: 'bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90',
        success: 'bg-success/15 text-success border border-success/25 shadow-sm hover:bg-success/25',
        warning: 'bg-warning/15 text-warning border border-warning/25 shadow-sm hover:bg-warning/25',
        menu: 'text-foreground hover:bg-primary/8 hover:text-primary focus-visible:ring-ring',
        bare: 'text-foreground hover:text-primary focus-visible:ring-ring',
        select:
          'text-foreground hover:bg-primary/8 focus-visible:ring-ring border border-border/50 px-2 py-1 rounded-lg',
        link: 'text-primary underline-offset-4 hover:underline p-0 h-auto',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 px-3 text-xs',
        lg: 'h-11 px-6',
        xl: 'h-12 px-8 text-base',
        icon: 'h-10 w-10',
        'icon-sm': 'h-7 w-7',
        'icon-lg': 'h-11 w-11',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

/* ── Spinner ───────────────────────────────────────────────── */

function ButtonSpinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )
}

/* ── Button ────────────────────────────────────────────────── */

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  /** Shows a spinner and disables the button */
  loading?: boolean
  /** Label shown next to spinner (defaults to children) */
  loadingText?: string
  /** Icon placed before children */
  leftIcon?: ReactNode
  /** Icon placed after children */
  rightIcon?: ReactNode
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant, size, type, loading, loadingText, leftIcon, rightIcon, children, disabled, ...props },
    ref
  ) => {
    const isDisabled = disabled || loading

    return (
      <button
        ref={ref}
        type={type ?? 'button'}
        disabled={isDisabled}
        aria-busy={loading ?? undefined}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      >
        {loading ? (
          <>
            <ButtonSpinner />
            {loadingText ?? children}
          </>
        ) : (
          <>
            {leftIcon && <span className="shrink-0">{leftIcon}</span>}
            {children}
            {rightIcon && <span className="shrink-0">{rightIcon}</span>}
          </>
        )}
      </button>
    )
  }
)
Button.displayName = 'Button'
