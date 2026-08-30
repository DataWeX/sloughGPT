'use client'

import { cn } from '../../lib/cn'

interface LoadingDotsProps {
  size?: 'sm' | 'default' | 'lg'
  className?: string
}

export function LoadingDots({ size = 'default', className }: LoadingDotsProps) {
  const dotSizes = { sm: 'w-1 h-1', default: 'w-1.5 h-1.5', lg: 'w-2 h-2' }
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn('inline-flex items-center gap-0.5', className)}
    >
      <span className={cn(dotSizes[size], 'bg-current rounded-full animate-bounce [animation-delay:0ms]')} />
      <span className={cn(dotSizes[size], 'bg-current rounded-full animate-bounce [animation-delay:150ms]')} />
      <span className={cn(dotSizes[size], 'bg-current rounded-full animate-bounce [animation-delay:300ms]')} />
    </span>
  )
}
