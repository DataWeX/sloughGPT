'use client'

import { type ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface AvatarProps {
  src?: string
  alt?: string
  fallback: string
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function Avatar({ src, alt, fallback, size = 'md', className }: AvatarProps) {
  const sizes = { sm: 'h-6 w-6 text-[10px]', md: 'h-8 w-8 text-xs', lg: 'h-10 w-10 text-sm' }

  return (
    <div className={cn("relative rounded-full overflow-hidden bg-primary/10 flex items-center justify-center", sizes[size], className)}>
      {src ? (
        <img src={src} alt={alt} className="h-full w-full object-cover" />
      ) : (
        <span className="font-medium text-primary">{fallback}</span>
      )}
    </div>
  )
}

interface AvatarGroupProps {
  avatars: { src?: string; alt?: string; fallback: string }[]
  max?: number
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function AvatarGroup({ avatars, max = 4, size = 'sm', className }: AvatarGroupProps) {
  const display = avatars.slice(0, max)
  const remaining = avatars.length - max
  const sizes = { sm: 'h-6 w-6 -ml-2', md: 'h-8 w-8 -ml-3', lg: 'h-10 w-10 -ml-4' }

  return (
    <div className={cn("flex items-center", className)}>
      {display.map((av, i) => (
        <Avatar key={i} src={av.src} alt={av.alt} fallback={av.fallback} size={size} className={cn(i > 0 && "border-2 border-background", sizes[size])} />
      ))}
      {remaining > 0 && (
        <div className={cn("rounded-full bg-muted flex items-center justify-center text-xs font-medium", sizes[size], "border-2 border-background")}>
          +{remaining}
        </div>
      )}
    </div>
  )
}

interface ProgressBarProps {
  value: number
  max?: number
  label?: string
  showValue?: boolean
  variant?: 'default' | 'success' | 'warning' | 'error'
  className?: string
}

export function ProgressBar({ value, max = 100, label, showValue = true, variant = 'default', className }: ProgressBarProps) {
  const percent = Math.min((value / max) * 100, 100)
  const variants = {
    default: "bg-primary",
    success: "bg-green-500",
    warning: "bg-yellow-500",
    error: "bg-red-500",
  }

  return (
    <div className={cn("space-y-1", className)}>
      {(label || showValue) && (
        <div className="flex justify-between text-xs">
          {label && <span className="text-muted-foreground">{label}</span>}
          {showValue && <span>{Math.round(percent)}%</span>}
        </div>
      )}
      <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", variants[variant])} style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function Spinner({ size = 'md', className }: SpinnerProps) {
  const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-8 h-8' }

  return (
    <svg className={cn("animate-spin", sizes[size], className)} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  )
}

interface DividerProps {
  label?: string
  className?: string
}

export function Divider({ label, className }: DividerProps) {
  if (label) {
    return (
      <div className={cn("flex items-center gap-3", className)}>
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-muted-foreground">{label}</span>
        <div className="flex-1 h-px bg-border" />
      </div>
    )
  }
  return <div className={cn("h-px bg-border", className)} />
}

interface CardDeckProps {
  title?: string
  description?: string
  footer?: ReactNode
  children: ReactNode
  className?: string
}

export function CardDeck({ title, description, footer, children, className }: CardDeckProps) {
  return (
    <div className={cn("border rounded-lg overflow-hidden", className)}>
      {(title || description) && (
        <div className="px-4 py-3 border-b">
          {title && <p className="font-medium text-sm">{title}</p>}
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
        </div>
      )}
      <div className="p-4">{children}</div>
      {footer && <div className="px-4 py-3 border-t bg-muted/30">{footer}</div>}
    </div>
  )
}

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: ReactNode
  className?: string
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center py-12 px-4 text-center", className)}>
      {icon && <div className="mb-4 text-muted-foreground">{icon}</div>}
      <p className="text-sm font-medium">{title}</p>
      {description && <p className="text-xs text-muted-foreground mt-1">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

interface SearchFieldProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  debounce?: number
  className?: string
}

export function SearchField({ value, onChange, placeholder = 'Search...', className }: SearchFieldProps) {
  return (
    <div className={cn("relative", className)}>
      <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full pl-8 pr-3 py-1.5 text-sm rounded-md border border-input bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
      />
    </div>
  )
}

interface PaginationProps {
  page: number
  total: number
  pageSize: number
  onChange: (page: number) => void
  className?: string
}

export function Pagination({ page, total, pageSize, onChange, className }: PaginationProps) {
  const totalPages = Math.ceil(total / pageSize)
  if (totalPages <= 1) return null

  return (
    <div className={cn("flex items-center justify-between", className)}>
      <span className="text-xs text-muted-foreground">
        Page {page} of {totalPages}
      </span>
      <div className="flex gap-1">
        <button onClick={() => onChange(page - 1)} disabled={page <= 1} className="px-2 py-1 text-xs rounded border disabled:opacity-50">
          Prev
        </button>
        <button onClick={() => onChange(page + 1)} disabled={page >= totalPages} className="px-2 py-1 text-xs rounded border disabled:opacity-50">
          Next
        </button>
      </div>
    </div>
  )
}
