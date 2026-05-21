import * as React from 'react'

import { cn } from '../../lib/cn'

export interface EmptyStateProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string
  description?: string
}

/** Centered empty chat / no data — stacks on narrow viewports. */
export function EmptyState({ className, title, description, children, ...props }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'str-safe-x flex flex-col items-center justify-center gap-3 px-4 py-10 text-center sm:py-14',
        className,
      )}
      role="status"
      aria-live="polite"
      {...props}
    >
      <div className="max-w-sm">
        <h2 className="text-lg font-semibold text-foreground sm:text-xl">{title}</h2>
        {description ? <p className="mt-2 text-sm text-muted-foreground">{description}</p> : null}
      </div>
      {children ? <div className="flex w-full max-w-xs flex-col gap-2 sm:max-w-sm">{children}</div> : null}
    </div>
  )
}
