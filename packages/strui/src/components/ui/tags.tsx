'use client'

import { type ReactNode, useState, useRef, useCallback, KeyboardEvent } from 'react'
import { cn } from '../../lib/cn'

/* ── Chip ───────────────────────────────────────────────────── */

interface ChipProps {
  label: string
  selected?: boolean
  onClick?: () => void
  icon?: ReactNode
  removable?: boolean
  onRemove?: () => void
  variant?: 'default' | 'outline' | 'solid' | 'success' | 'warning' | 'error'
  size?: 'xs' | 'sm' | 'default'
  disabled?: boolean
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
  disabled,
  className,
}: ChipProps) {
  const Tag = onClick ? 'button' : 'span'

  const baseStyles = 'inline-flex items-center gap-1 rounded-full font-medium transition-all duration-150 select-none'

  const variantStyles = {
    default: selected
      ? 'bg-primary text-primary-foreground shadow-sm'
      : 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
    outline: selected
      ? 'border border-primary bg-primary/10 text-primary'
      : 'border border-border bg-transparent text-foreground hover:border-primary/40 hover:text-primary',
    solid: selected
      ? 'bg-primary text-primary-foreground'
      : 'bg-muted text-muted-foreground hover:bg-muted/70 hover:text-foreground',
    success: selected
      ? 'bg-success text-white shadow-sm'
      : 'bg-success/15 text-success hover:bg-success/25',
    warning: selected
      ? 'bg-warning text-white shadow-sm'
      : 'bg-warning/15 text-warning hover:bg-warning/25',
    error: selected
      ? 'bg-destructive text-white shadow-sm'
      : 'bg-destructive/15 text-destructive hover:bg-destructive/25',
  }

  const sizeStyles = {
    xs: 'px-1.5 py-0.5 text-[10px]',
    sm: 'px-2 py-0.5 text-[11px]',
    default: 'px-2.5 py-1 text-xs',
  }

  return (
    <Tag
      type={onClick ? 'button' : undefined}
      onClick={disabled ? undefined : onClick}
      aria-pressed={onClick ? selected : undefined}
      aria-disabled={disabled}
      disabled={onClick && disabled ? true : undefined}
      className={cn(
        baseStyles,
        variantStyles[variant],
        sizeStyles[size],
        onClick && !disabled && 'cursor-pointer',
        disabled && 'opacity-40 cursor-not-allowed',
        className,
      )}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      {label}
      {removable && (
        <button
          type="button"
          aria-label={`Remove ${label}`}
          onClick={(e) => {
            e.stopPropagation()
            if (!disabled) onRemove?.()
          }}
          className={cn(
            'ml-0.5 flex items-center justify-center rounded-full transition-opacity',
            'hover:opacity-70 focus:outline-none focus:ring-1 focus:ring-primary/50',
            disabled && 'pointer-events-none',
          )}
        >
          <svg className="h-2.5 w-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </Tag>
  )
}

/* ── Chips (multi-select) ───────────────────────────────────── */

interface ChipsProps {
  value: string[]
  onChange: (value: string[]) => void
  options: { value: string; label: string; icon?: ReactNode; disabled?: boolean }[]
  max?: number
  variant?: ChipProps['variant']
  size?: ChipProps['size']
  className?: string
}

export function Chips({ value, onChange, options, max, variant = 'default', size = 'default', className }: ChipsProps) {
  const toggle = (optValue: string) => {
    if (value.includes(optValue)) {
      onChange(value.filter((v) => v !== optValue))
    } else if (!max || value.length < max) {
      onChange([...value, optValue])
    }
  }

  return (
    <div className={cn('flex flex-wrap gap-1.5', className)} role="group">
      {options.map((opt) => (
        <Chip
          key={opt.value}
          label={opt.label}
          selected={value.includes(opt.value)}
          onClick={() => toggle(opt.value)}
          icon={opt.icon}
          variant={variant}
          size={size}
          disabled={opt.disabled || (!value.includes(opt.value) && max !== undefined && value.length >= max)}
        />
      ))}
    </div>
  )
}

/* ── Tag Input ──────────────────────────────────────────────── */

interface TagInputProps {
  value: string[]
  onChange: (value: string[]) => void
  placeholder?: string
  /** Characters that trigger tag creation. Default: [',', 'Enter'] */
  delimiters?: string[]
  maxTags?: number
  disabled?: boolean
  className?: string
}

export function TagInput({
  value,
  onChange,
  placeholder = 'Add tag…',
  delimiters = ['Enter', ','],
  maxTags,
  disabled,
  className,
}: TagInputProps) {
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const addTag = useCallback(
    (raw: string) => {
      const trimmed = raw.trim().replace(/,$/, '')
      if (!trimmed) return
      if (value.includes(trimmed)) { setInput(''); return }
      if (maxTags !== undefined && value.length >= maxTags) return
      onChange([...value, trimmed])
      setInput('')
    },
    [value, onChange, maxTags],
  )

  const remove = (tag: string) => onChange(value.filter((t) => t !== tag))

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (delimiters.includes(e.key)) {
      e.preventDefault()
      addTag(input)
    }
    // Remove last tag on Backspace when input is empty
    if (e.key === 'Backspace' && !input && value.length > 0) {
      remove(value[value.length - 1])
    }
  }

  const atMax = maxTags !== undefined && value.length >= maxTags

  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-1.5 min-h-10 px-2 py-1.5 rounded-lg border border-input bg-background',
        'transition-[border-color,box-shadow] duration-200',
        'focus-within:ring-2 focus-within:ring-primary/40 focus-within:ring-offset-2 focus-within:border-primary/60',
        disabled && 'cursor-not-allowed opacity-40',
        className,
      )}
      onClick={() => inputRef.current?.focus()}
    >
      {value.map((tag) => (
        <Chip
          key={tag}
          label={tag}
          removable
          onRemove={() => !disabled && remove(tag)}
          size="sm"
          variant="solid"
          selected
        />
      ))}
      {!atMax && (
        <input
          ref={inputRef}
          type="text"
          value={input}
          disabled={disabled}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => addTag(input)}
          placeholder={value.length === 0 ? placeholder : ''}
          className="flex-1 min-w-[80px] bg-transparent outline-none text-sm placeholder:text-muted-foreground disabled:cursor-not-allowed"
        />
      )}
    </div>
  )
}
