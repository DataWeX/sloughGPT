import { XIcon } from '../../lib/icons'
import { cn } from '../../lib/cn'

export interface AttachmentChipProps {
  name: string
  onRemove?: () => void
  className?: string
}

/** File name pill above the composer (images, PDFs for multimodal). */
export function AttachmentChip({ name, onRemove, className }: AttachmentChipProps) {
  return (
    <span
      className={cn(
        'inline-flex max-w-full items-center gap-1 rounded-none border border-border bg-card px-2 py-1 text-xs text-foreground shadow-sm',
        className,
      )}
    >
      <span className="min-w-0 truncate font-mono">{name}</span>
      {onRemove ? (
        <button
          type="button"
          className="str-touch-target flex h-7 w-7 shrink-0 items-center justify-center sl-hover-primary"
          onClick={onRemove}
          aria-label={`Remove ${name}`}
        >
          <XIcon />
        </button>
      ) : null}
    </span>
  )
}
