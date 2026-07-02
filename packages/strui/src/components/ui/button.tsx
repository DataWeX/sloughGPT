import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '../../lib/cn'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-none text-sm font-medium transition-all duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 active:scale-[0.99]',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground shadow-sm hover:opacity-90',
        secondary:
          'border border-border bg-secondary text-secondary-foreground shadow-sm hover:bg-primary/10 hover:text-primary hover:border-primary/30',
        ghost: 'hover:bg-primary/10 hover:text-primary',
        outline:
          'border border-border bg-transparent text-foreground shadow-sm hover:bg-primary/10 hover:text-primary hover:border-primary/30',
        destructive: 'bg-destructive text-destructive-foreground shadow-sm hover:opacity-90',
        menu: 'text-foreground hover:bg-primary/10 hover:text-primary focus-visible:ring-primary/50',
        bare: 'text-foreground hover:text-primary focus-visible:ring-primary/50',
        select: 'text-foreground hover:bg-primary/10 focus-visible:ring-primary/50 border border-border/50 px-2 py-1 rounded',
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
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, type, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return (
      <Comp
        className={cn(buttonVariants({ variant, size }), className)}
        ref={ref}
        {...props}
        {...(!asChild ? { type: type ?? 'button' } : {})}
      />
    )
  }
)
Button.displayName = 'Button'

export { Button, buttonVariants }
