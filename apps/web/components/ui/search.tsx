'use client'

import { useState, useRef, useEffect } from 'react'
import { cn } from '@/lib/cn'

interface SearchBoxProps {
  value?: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}

export function SearchBox({ value = '', onChange, placeholder = 'Search...', className }: SearchBoxProps) {
  return (
    <div className={cn("relative", className)}>
      <svg
        className="absolute left-2 top-1/2 -translate-y-1/2 h-2.5 w-2.5 text-muted-foreground pointer-events-none"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full pl-6 pr-2 py-1 text-xs rounded border border-input bg-background focus:outline-none focus:ring-1 focus:ring-primary/50"
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className="absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground hover:text-foreground"
          aria-label="Clear search"
        >
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  )
}

interface SectionTabsProps {
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string; icon?: React.ReactNode; count?: number }[]
  className?: string
}

export function SectionTabs({ value, onChange, options, className }: SectionTabsProps) {
  return (
    <div className={cn("flex items-center gap-1 p-0.5 bg-muted/50 rounded-md", className)}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            "flex-1 flex items-center justify-center gap-1 py-1 px-1.5 rounded text-[10px] font-medium transition-colors",
            value === opt.value
              ? "bg-background shadow-sm text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          {opt.icon}
          {opt.label}
          {opt.count !== undefined && opt.count > 0 && (
            <span className="text-[10px]">{opt.count}</span>
          )}
        </button>
      ))}
    </div>
  )
}

interface ActionButtonProps {
  icon: React.ReactNode
  label: string
  onClick: () => void
  variant?: 'default' | 'ghost' | 'destructive'
  className?: string
}

export function ActionButton({ icon, label, onClick, variant = 'default', className }: ActionButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center justify-center gap-1 py-1 px-2 rounded text-[10px] font-medium transition-colors",
        variant === 'default' && "bg-primary text-primary-foreground hover:bg-primary/90",
        variant === 'ghost' && "text-muted-foreground hover:text-foreground hover:bg-accent",
        variant === 'destructive' && "text-destructive hover:bg-destructive/10",
        className
      )}
    >
      {icon}
      {label}
    </button>
  )
}

interface EmptyStateProps {
  message?: string
  action?: { label: string; onClick: () => void }
}

export function EmptyState({ message = 'No items', action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
      <p className="text-xs text-muted-foreground">{message}</p>
      {action && (
        <button
          onClick={action.onClick}
          className="mt-2 text-xs text-primary hover:underline"
        >
          {action.label}
        </button>
      )}
    </div>
  )
}