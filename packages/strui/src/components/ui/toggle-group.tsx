'use client'

import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from 'react'
import { cn } from '../../lib/cn'
import { cva, type VariantProps } from 'class-variance-authority'

/* ── Context ────────────────────────────────────────────────── */

interface ToggleGroupContextValue {
  type: 'single' | 'multiple'
  value: string | string[]
  onValueChange: (itemValue: string) => void
  variant?: 'default' | 'outline' | 'pills'
  size?: 'sm' | 'default' | 'lg'
  disabled?: boolean
}

const ToggleGroupContext = createContext<ToggleGroupContextValue | null>(null)

function useToggleGroupContext() {
  const ctx = useContext(ToggleGroupContext)
  if (!ctx) throw new Error('ToggleGroupItem must be used within <ToggleGroup>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────── */

interface ToggleGroupRootProps {
  type?: 'single' | 'multiple'
  value?: string | string[]
  defaultValue?: string | string[]
  onValueChange?: (value: string | string[]) => void
  className?: string
  children: ReactNode
  variant?: 'default' | 'outline' | 'pills'
  size?: 'sm' | 'default' | 'lg'
  disabled?: boolean
}

function ToggleGroup({
  type = 'single',
  value: controlledValue,
  defaultValue,
  onValueChange,
  className,
  children,
  variant = 'default',
  size = 'default',
  disabled,
}: ToggleGroupRootProps) {
  const fallback = defaultValue ?? (type === 'single' ? '' : [])
  const [internalValue, setInternalValue] = useState<string | string[]>(fallback)
  const isControlled = controlledValue !== undefined
  const value = isControlled ? controlledValue! : internalValue

  const handleValueChange = useCallback(
    (itemValue: string) => {
      if (disabled) return
      let next: string | string[]
      if (type === 'single') {
        next = value === itemValue ? '' : itemValue
      } else {
        const arr = Array.isArray(value) ? value : []
        next = arr.includes(itemValue) ? arr.filter((v) => v !== itemValue) : [...arr, itemValue]
      }
      if (!isControlled) setInternalValue(next)
      onValueChange?.(next)
    },
    [type, value, isControlled, disabled, onValueChange],
  )

  const containerStyles = {
    default: 'inline-flex h-10 items-center justify-center gap-0.5 rounded-lg border border-border bg-muted/50 p-1',
    outline: 'inline-flex items-center justify-center gap-1',
    pills: 'inline-flex flex-wrap items-center gap-1',
  }

  return (
    <ToggleGroupContext.Provider value={{ type, value, onValueChange: handleValueChange, variant, size, disabled }}>
      <div
        role={type === 'single' ? 'radiogroup' : 'group'}
        aria-disabled={disabled}
        className={cn(containerStyles[variant], className)}
      >
        {children}
      </div>
    </ToggleGroupContext.Provider>
  )
}

/* ── Item ───────────────────────────────────────────────────── */

const itemVariants = cva(
  [
    'inline-flex items-center justify-center gap-1.5 whitespace-nowrap font-medium transition-all duration-150',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
    'disabled:pointer-events-none disabled:opacity-40',
  ].join(' '),
  {
    variants: {
      variant: {
        default: 'rounded-md',
        outline: 'rounded-lg border border-border',
        pills: 'rounded-full border',
      },
      size: {
        sm: 'px-2 py-1 text-xs',
        default: 'px-3 py-1.5 text-sm',
        lg: 'px-4 py-2 text-sm',
      },
      active: {
        true: '',
        false: '',
      },
    },
    compoundVariants: [
      { variant: 'default', active: true, class: 'bg-background text-foreground shadow-sm' },
      { variant: 'default', active: false, class: 'text-muted-foreground hover:text-foreground hover:bg-background/50' },
      { variant: 'outline', active: true, class: 'border-primary/50 bg-primary/10 text-primary' },
      { variant: 'outline', active: false, class: 'border-border bg-transparent text-muted-foreground hover:text-foreground hover:bg-muted/50' },
      { variant: 'pills', active: true, class: 'border-primary bg-primary text-primary-foreground shadow-sm' },
      { variant: 'pills', active: false, class: 'border-border bg-transparent text-muted-foreground hover:border-primary/40 hover:text-foreground' },
    ],
    defaultVariants: { variant: 'default', size: 'default', active: false },
  }
)

interface ToggleGroupItemProps extends HTMLAttributes<HTMLButtonElement> {
  value: string
  disabled?: boolean
}

const ToggleGroupItem = forwardRef<HTMLButtonElement, ToggleGroupItemProps>(
  ({ value: itemValue, className, children, disabled: itemDisabled, ...props }, ref) => {
    const { type, value, onValueChange, variant = 'default', size = 'default', disabled: groupDisabled } = useToggleGroupContext()
    const isActive = type === 'single' ? value === itemValue : Array.isArray(value) && value.includes(itemValue)
    const isDisabled = itemDisabled || groupDisabled

    return (
      <button
        ref={ref}
        type="button"
        role={type === 'single' ? 'radio' : 'checkbox'}
        aria-checked={isActive}
        aria-disabled={isDisabled}
        disabled={isDisabled}
        data-state={isActive ? 'on' : 'off'}
        onClick={() => onValueChange(itemValue)}
        className={cn(itemVariants({ variant, size, active: isActive }), className)}
        {...props}
      >
        {children}
      </button>
    )
  },
)
ToggleGroupItem.displayName = 'ToggleGroupItem'

export { ToggleGroup, ToggleGroupItem }
