'use client'

import { type ReactNode, useState } from 'react'
import { cn } from '@/lib/cn'

interface ChipProps {
  label: string
  selected?: boolean
  onClick?: () => void
  icon?: ReactNode
  removable?: boolean
  onRemove?: () => void
  variant?: 'default' | 'outline'
  size?: 'sm' | 'default'
  className?: string
}

export function Chip({
  label,
  selected = false,
  onClick,
  icon,
  removable = false,
  onRemove,
  variant = 'default',
  size = 'default',
  className,
}: ChipProps) {
  const Component = onClick ? 'button' : 'span'

  return (
    <Component
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1 rounded-full font-medium transition-colors",
        selected && "bg-primary text-primary-foreground",
        !selected && variant === 'default' && "bg-primary/10 text-primary",
        !selected && variant === 'outline' && "border border-border bg-transparent",
        size === 'sm' && "px-2 py-0.5 text-[10px]",
        size === 'default' && "px-2.5 py-1 text-xs",
        onClick && "cursor-pointer hover:opacity-80",
        className
      )}
    >
      {icon}
      {label}
      {removable && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onRemove?.() }}
          className="ml-0.5 hover:opacity-70"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </Component>
  )
}

interface ChipsProps {
  value: string[]
  onChange: (value: string[]) => void
  options: { value: string; label: string; icon?: ReactNode }[]
  max?: number
  className?: string
}

export function Chips({ value, onChange, options, max, className }: ChipsProps) {
  const toggle = (optValue: string) => {
    if (value.includes(optValue)) {
      onChange(value.filter(v => v !== optValue))
    } else if (!max || value.length < max) {
      onChange([...value, optValue])
    }
  }

  return (
    <div className={cn("flex flex-wrap gap-1.5", className)}>
      {options.map((opt) => (
        <Chip
          key={opt.value}
          label={opt.label}
          selected={value.includes(opt.value)}
          onClick={() => toggle(opt.value)}
          icon={opt.icon}
        />
      ))}
    </div>
  )
}

interface BadgeProps {
  label: string
  variant?: 'default' | 'success' | 'warning' | 'error' | 'outline'
  size?: 'sm' | 'default'
  className?: string
}

export function Badge({
  label,
  variant = 'default',
  size = 'default',
  className,
}: BadgeProps) {
  const variants = {
    default: "bg-primary/10 text-primary",
    success: "bg-success/10 text-success",
    warning: "bg-warning/10 text-warning",
    error: "bg-destructive/10 text-destructive",
    outline: "border border-border bg-transparent",
  }

  return (
    <span className={cn(
      "inline-flex items-center rounded-full font-medium",
      variants[variant],
      size === 'sm' && "px-1.5 py-0.5 text-[10px]",
      size === 'default' && "px-2 py-0.5 text-xs",
      className
    )}>
      {label}
    </span>
  )
}

interface TagInputProps {
  value: string[]
  onChange: (value: string[]) => void
  placeholder?: string
  className?: string
}

export function TagInput({ value, onChange, placeholder = 'Add tag...', className }: TagInputProps) {
  const [input, setInput] = useState('')

  const add = () => {
    const trimmed = input.trim()
    if (trimmed && !value.includes(trimmed)) {
      onChange([...value, trimmed])
      setInput('')
    }
  }

  const remove = (tag: string) => {
    onChange(value.filter(t => t !== tag))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      add()
    }
  }

  return (
    <div className={cn("flex flex-wrap gap-1.5 p-2 border rounded-lg min-h-[40px]", className)}>
      {value.map((tag) => (
        <Chip
          key={tag}
          label={tag}
          removable
          onRemove={() => remove(tag)}
        />
      ))}
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={value.length === 0 ? placeholder : ''}
        className="flex-1 min-w-[60px] bg-transparent outline-none text-sm placeholder:text-muted-foreground"
      />
    </div>
  )
}