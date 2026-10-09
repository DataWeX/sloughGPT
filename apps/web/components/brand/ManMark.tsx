import { cn } from '@/lib/utils'

export const MAN_NAME = 'Man'
export const MAN_FULL_NAME = 'Man - AI Platform'
export const MAN_TAGLINE = 'Your personal AI — trained on your own data.'

export function ManMark({ className, letter = 'M' }: { className?: string; letter?: string }) {
  return (
    <div
      aria-hidden
      data-testid="man-mark"
      className={cn(
        'flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 font-mono text-[11px] font-bold leading-none tracking-tight text-primary-foreground shadow-sm',
        className,
      )}
    >
      {letter}
    </div>
  )
}

export function ManLockup({ className }: { className?: string }) {
  return (
    <div data-testid="brand-lockup" className={cn('flex items-center gap-3', className)}>
      <ManMark className="h-9 w-9 rounded-xl text-sm" />
      <div className="min-w-0">
        <div className="text-sm font-semibold text-foreground">{MAN_NAME}</div>
        <div className="text-xs text-muted-foreground">{MAN_TAGLINE}</div>
      </div>
    </div>
  )
}
