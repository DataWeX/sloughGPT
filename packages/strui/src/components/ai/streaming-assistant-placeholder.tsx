import { Skeleton } from '../composed/skeleton'
import { cn } from '../../lib/cn'

export interface StreamingAssistantPlaceholderProps {
  className?: string
  /** Number of skeleton lines (default 3). */
  lines?: number
}

/** Placeholder while assistant message is streaming (matches **MessageBubble** width). */
export function StreamingAssistantPlaceholder({
  className,
  lines = 3,
}: StreamingAssistantPlaceholderProps) {
  return (
    <div
      className={cn(
        'mr-auto max-w-[var(--chat-thread-max)] border border-border bg-card/60 px-3 py-3 shadow-sm sm:px-4',
        className,
      )}
      aria-hidden
    >
      <div className="flex flex-col gap-2">
        {Array.from({ length: lines }).map((_, i) => (
          <Skeleton
            key={i}
            className={cn('h-3', i === 0 ? 'w-[92%]' : i === 1 ? 'w-[78%]' : 'w-[64%]')}
          />
        ))}
      </div>
    </div>
  )
}
