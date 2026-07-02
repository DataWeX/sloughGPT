import { cva, type VariantProps } from 'class-variance-authority'
import type { HTMLAttributes, ReactNode } from 'react'

import { cn } from '../../lib/cn'

const bannerVariants = cva(
  'flex flex-col gap-2 rounded-none border px-4 py-3 text-sm sm:flex-row sm:items-center sm:justify-between sm:gap-4',
  {
    variants: {
      variant: {
        info: 'border-border bg-muted/40 text-foreground',
        success: 'border-success/40 bg-success/10 text-foreground',
        warning: 'border-warning/40 bg-warning/10 text-foreground',
        error: 'border-destructive/40 bg-destructive/10 text-foreground',
      },
    },
    defaultVariants: {
      variant: 'info',
    },
  },
)

export interface InlineBannerProps
  extends HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof bannerVariants> {
  title: string
  description?: string
  /** Primary action (e.g. retry). */
  action?: ReactNode
}

/** Non-blocking inline alert for API status, tool errors, or quota warnings. */
export function InlineBanner({
  className,
  variant,
  title,
  description,
  action,
  ...props
}: InlineBannerProps) {
  return (
    <div className={cn(bannerVariants({ variant }), className)} role="status" {...props}>
      <div className="min-w-0">
        <p className="font-medium">{title}</p>
        {description ? <p className="mt-0.5 text-muted-foreground">{description}</p> : null}
      </div>
      {action ? <div className="flex shrink-0 flex-wrap gap-2">{action}</div> : null}
    </div>
  )
}

export { bannerVariants }
