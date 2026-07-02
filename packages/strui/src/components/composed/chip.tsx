import { cva, type VariantProps } from 'class-variance-authority'
import type { HTMLAttributes, ReactNode } from 'react'

import { XIcon } from '../../lib/icons'
import { cn } from '../../lib/cn'

const chipVariants = cva(
  'inline-flex max-w-full items-center gap-1 rounded-none border px-2 py-1 text-xs font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'border-border bg-muted/50 text-foreground',
        primary: 'border-primary/35 bg-primary/10 text-foreground',
        outline: 'border-border bg-transparent text-muted-foreground',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

export interface ChipProps
  extends Omit<HTMLAttributes<HTMLSpanElement>, 'children'>,
    VariantProps<typeof chipVariants> {
  children: ReactNode
  onRemove?: () => void
}

/** Generic filter / entity chip — use **AttachmentChip** for file-specific UX. */
export function Chip({ className, variant, children, onRemove, ...props }: ChipProps) {
  return (
    <span className={cn(chipVariants({ variant }), className)} {...props}>
      <span className="min-w-0 truncate">{children}</span>
      {onRemove ? (
        <button
          type="button"
          className="str-touch-target -mr-0.5 flex h-6 w-6 shrink-0 items-center justify-center sl-hover-primary"
          onClick={onRemove}
          aria-label="Remove"
        >
          <XIcon />
        </button>
      ) : null}
    </span>
  )
}

export { chipVariants }
