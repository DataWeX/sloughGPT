'use client'

import { forwardRef, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface SliderProps {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  label?: string
  showValue?: boolean
  formatValue?: (value: number) => string
  className?: string
}

export function Slider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  showValue = true,
  formatValue,
  className,
}: SliderProps) {
  const displayValue = formatValue ? formatValue(value) : value.toString()

  return (
    <div className={cn("space-y-2", className)}>
      {label && (
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">{label}</label>
          {showValue && <span className="text-sm text-muted-foreground">{displayValue}</span>}
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-2 w-full cursor-pointer accent-primary"
      />
    </div>
  )
}

interface RangeSliderProps {
  value: [number, number]
  onChange: (value: [number, number]) => void
  min?: number
  max?: number
  step?: number
  label?: string
  formatValue?: (value: number) => string
  className?: string
}

export function RangeSlider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  formatValue,
  className,
}: RangeSliderProps) {
  const format = (v: number) => formatValue ? formatValue(v) : v.toString()

  return (
    <div className={cn("space-y-2", className)}>
      {label && (
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">{label}</label>
          <span className="text-sm text-muted-foreground">
            {format(value[0])} - {format(value[1])}
          </span>
        </div>
      )}
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[0]}
          onChange={(e) => onChange([parseFloat(e.target.value), value[1]])}
          className="flex-1 h-2 cursor-pointer accent-primary"
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[1]}
          onChange={(e) => onChange([value[0], parseFloat(e.target.value)])}
          className="flex-1 h-2 cursor-pointer accent-primary"
        />
      </div>
    </div>
  )
}

interface ToggleProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label?: string
  description?: string
  disabled?: boolean
  className?: string
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled,
  className,
}: ToggleProps) {
  return (
    <label className={cn("flex items-center justify-between gap-4 cursor-pointer", disabled && "opacity-50 cursor-not-allowed", className)}>
      {(label || description) && (
        <div>
          {label && <p className="text-sm font-medium">{label}</p>}
          {description && <p className="text-xs text-muted-foreground">{description}</p>}
        </div>
      )}
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={cn(
          "relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors",
          checked ? "bg-primary" : "bg-muted"
        )}
      >
        <span
          className={cn(
            "pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow-lg ring-0 transition-transform",
            checked ? "translate-x-4" : "translate-x-0"
          )}
        />
      </button>
    </label>
  )
}

interface FieldGroupProps {
  label?: string
  description?: string
  error?: string
  children: ReactNode
  className?: string
}

export function FieldGroup({ label, description, error, children, className }: FieldGroupProps) {
  return (
    <div className={cn("space-y-2", className)}>
      {label && <label className="text-sm font-medium">{label}</label>}
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
      {children}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  )
}

interface ToggleGroupOption {
  value: string
  label: string
  icon?: ReactNode
}

interface ToggleGroupProps {
  value: string
  onChange: (value: string) => void
  options: ToggleGroupOption[]
  className?: string
}

export function ToggleGroup({ value, onChange, options, className }: ToggleGroupProps) {
  return (
    <div className={cn("flex items-center gap-1 p-0.5 bg-muted/50 rounded-lg", className)}>
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={cn(
            "flex-1 flex items-center justify-center gap-1 py-1 px-2 rounded-md text-[10px] font-medium transition-colors",
            value === opt.value
              ? "bg-background shadow-sm text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          {opt.icon}
          {opt.label}
        </button>
      ))}
    </div>
  )
}

interface TabsProps {
  value: string
  onChange: (value: string) => void
  tabs: { value: string; label: string; count?: number }[]
  className?: string
}

export function Tabs({ value, onChange, tabs, className }: TabsProps) {
  return (
    <div className={cn("flex items-center gap-1 p-0.5 bg-muted/50 rounded-lg", className)}>
      {tabs.map((tab) => (
        <button
          key={tab.value}
          type="button"
          onClick={() => onChange(tab.value)}
          className={cn(
            "flex-1 flex items-center justify-center gap-1 py-1 px-2 rounded-md text-[10px] font-medium transition-colors",
            value === tab.value
              ? "bg-background shadow-sm text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          {tab.label}
          {tab.count !== undefined && (
            <span className="text-[10px]">{tab.count}</span>
          )}
        </button>
      ))}
    </div>
  )
}