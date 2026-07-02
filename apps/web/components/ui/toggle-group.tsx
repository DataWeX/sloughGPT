'use client'

import { createContext, forwardRef, useCallback, useContext, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

/* ── Context ────────────────────────────────────────────────────── */

interface ToggleGroupContextValue {
  type: 'single' | 'multiple'
  value: string | string[]
  onValueChange: (itemValue: string) => void
}

const ToggleGroupContext = createContext<ToggleGroupContextValue | null>(null)

function useToggleGroupContext() {
  const ctx = useContext(ToggleGroupContext)
  if (!ctx) throw new Error('ToggleGroupItem must be used within <ToggleGroup>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────────── */

interface ToggleGroupRootProps {
  type?: 'single' | 'multiple'
  value?: string | string[]
  defaultValue?: string | string[]
  onValueChange?: (value: string | string[]) => void
  className?: string
  children: ReactNode
}

function ToggleGroup({
  type = 'single',
  value: controlledValue,
  defaultValue = type === 'single' ? '' : [],
  onValueChange,
  className,
  children,
}: ToggleGroupRootProps) {
  const getValue = () => controlledValue ?? defaultValue

  const handleValueChange = useCallback(
    (itemValue: string) => {
      const current = getValue()
      if (type === 'single') {
        onValueChange?.(itemValue)
      } else {
        const arr = Array.isArray(current) ? current : []
        const next = arr.includes(itemValue)
          ? arr.filter((v) => v !== itemValue)
          : [...arr, itemValue]
        onValueChange?.(next)
      }
    },
    [type, controlledValue, defaultValue, onValueChange],
  )

  return (
    <ToggleGroupContext.Provider value={{ type, value: getValue(), onValueChange: handleValueChange }}>
      <div
        role="group"
        className={cn(
          'inline-flex h-10 items-center justify-center gap-1 rounded-lg border border-border bg-muted/50 p-1 text-muted-foreground',
          className,
        )}
      >
        {children}
      </div>
    </ToggleGroupContext.Provider>
  )
}

/* ── Item ───────────────────────────────────────────────────────── */

interface ToggleGroupItemProps {
  value: string
  className?: string
  children: ReactNode
  disabled?: boolean
}

const ToggleGroupItem = forwardRef<HTMLButtonElement, ToggleGroupItemProps>(
  ({ value: itemValue, className, children, disabled, ...props }, ref) => {
    const { type, value, onValueChange } = useToggleGroupContext()
    const isActive = type === 'single' ? value === itemValue : Array.isArray(value) && value.includes(itemValue)

    return (
      <button
        ref={ref}
        type="button"
        role={type === 'single' ? 'radio' : 'checkbox'}
        aria-checked={isActive}
        disabled={disabled}
        onClick={() => onValueChange(itemValue)}
        className={cn(
          'inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
          'disabled:pointer-events-none disabled:opacity-50',
          isActive ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
          className,
        )}
        {...props}
      >
        {children}
      </button>
    )
  },
)
ToggleGroupItem.displayName = 'ToggleGroupItem'

export { ToggleGroup, ToggleGroupItem }
