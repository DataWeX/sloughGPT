'use client'

import { cn } from '../../lib/cn'

interface AvatarProps {
  src?: string
  alt?: string
  fallback: string
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  className?: string
}

export function Avatar({ src, alt, fallback, size = 'md', className }: AvatarProps) {
  const sizes = {
    xs: 'h-5 w-5 text-[9px]',
    sm: 'h-6 w-6 text-[10px]',
    md: 'h-8 w-8 text-xs',
    lg: 'h-10 w-10 text-sm',
    xl: 'h-12 w-12 text-base',
  }

  return (
    <div
      className={cn(
        'relative rounded-full overflow-hidden bg-primary/10 flex items-center justify-center shrink-0 ring-1 ring-border/40',
        sizes[size],
        className,
      )}
    >
      {src ? (
        <img src={src} alt={alt ?? fallback} className="h-full w-full object-cover" />
      ) : (
        <span className="font-semibold text-primary select-none">{fallback.slice(0, 2).toUpperCase()}</span>
      )}
    </div>
  )
}

interface AvatarGroupProps {
  avatars: { src?: string; alt?: string; fallback: string }[]
  max?: number
  size?: 'xs' | 'sm' | 'md' | 'lg'
  className?: string
}

export function AvatarGroup({ avatars, max = 4, size = 'sm', className }: AvatarGroupProps) {
  const display = avatars.slice(0, max)
  const remaining = avatars.length - max

  const offsetMap = { xs: '-ml-1.5', sm: '-ml-2', md: '-ml-3', lg: '-ml-4' }
  const offset = offsetMap[size]

  return (
    <div className={cn('flex items-center', className)}>
      {display.map((av, i) => (
        <Avatar
          key={i}
          src={av.src}
          alt={av.alt}
          fallback={av.fallback}
          size={size}
          className={cn(i > 0 && offset, 'border-2 border-background')}
        />
      ))}
      {remaining > 0 && (
        <div
          className={cn(
            'rounded-full bg-muted flex items-center justify-center text-[10px] font-medium text-muted-foreground border-2 border-background',
            offset,
            size === 'xs' ? 'h-5 w-5' : size === 'sm' ? 'h-6 w-6' : size === 'md' ? 'h-8 w-8' : 'h-10 w-10',
          )}
        >
          +{remaining}
        </div>
      )}
    </div>
  )
}
