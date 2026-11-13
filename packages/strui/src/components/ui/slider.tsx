'use client'

import { forwardRef, useCallback, useRef, useState, type InputHTMLAttributes } from 'react'
import { cn } from '../../lib/cn'
import { inputFieldClassName } from './input'

/* ── Slider ─────────────────────────────────────────────────── */

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange' | 'value' | 'defaultValue' | 'size'> {
  value?: number[]
  defaultValue?: number[]
  min?: number
  max?: number
  step?: number
  label?: string
  /** Show current value next to label */
  showValue?: boolean
  /** Format the displayed value (e.g. `v => v + '%'`) */
  formatValue?: (value: number) => string
  onValueChange?: (value: number[]) => void
  size?: 'sm' | 'default' | 'lg'
}

const Slider = forwardRef<HTMLInputElement, SliderProps>(
  (
    {
      className,
      value,
      defaultValue,
      min = 0,
      max = 100,
      step = 1,
      onValueChange,
      disabled,
      label,
      showValue = false,
      formatValue,
      size = 'default',
      id,
      ...props
    },
    ref,
  ) => {
    const [internalValue, setInternalValue] = useState(defaultValue?.[0] ?? 0)
    const isControlled = value !== undefined
    const currentValue = isControlled ? (value?.[0] ?? 0) : internalValue

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const next = Number(e.target.value)
      if (!isControlled) setInternalValue(next)
      onValueChange?.([next])
    }

    const trackHeights = { sm: 'h-1', default: 'h-2', lg: 'h-3' }
    const thumbSizes = {
      sm: '[&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3',
      default: '[&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4',
      lg: '[&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5 [&::-moz-range-thumb]:h-5 [&::-moz-range-thumb]:w-5',
    }

    const displayValue = formatValue ? formatValue(currentValue) : String(currentValue)
    const fillPct = ((currentValue - min) / (max - min)) * 100

    const input = (
      <div className="relative flex items-center">
        {/* Filled track behind native input */}
        <div
          className={cn('absolute left-0 rounded-full bg-primary pointer-events-none', trackHeights[size])}
          style={{ width: `${fillPct}%` }}
        />
        <input
          ref={ref}
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={currentValue}
          disabled={disabled}
          onChange={handleChange}
          className={cn(
            'w-full cursor-pointer appearance-none bg-muted rounded-full',
            trackHeights[size],
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            // Thumb styling
            '[&::-webkit-slider-thumb]:appearance-none',
            '[&::-webkit-slider-thumb]:rounded-full',
            '[&::-webkit-slider-thumb]:bg-primary',
            '[&::-webkit-slider-thumb]:border-2',
            '[&::-webkit-slider-thumb]:border-background',
            '[&::-webkit-slider-thumb]:shadow-md',
            '[&::-webkit-slider-thumb]:transition-transform',
            '[&::-webkit-slider-thumb]:duration-100',
            '[&::-webkit-slider-thumb]:hover:scale-110',
            '[&::-moz-range-thumb]:rounded-full',
            '[&::-moz-range-thumb]:bg-primary',
            '[&::-moz-range-thumb]:border-2',
            '[&::-moz-range-thumb]:border-background',
            '[&::-moz-range-thumb]:shadow-md',
            thumbSizes[size],
            className,
          )}
          {...props}
        />
      </div>
    )

    if (label || showValue) {
      return (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            {label && (
              <label htmlFor={id} className="text-xs font-medium text-muted-foreground">
                {label}
              </label>
            )}
            {showValue && (
              <span className="text-xs font-semibold text-foreground tabular-nums">{displayValue}</span>
            )}
          </div>
          {input}
        </div>
      )
    }

    return input
  },
)
Slider.displayName = 'Slider'

/* ── Range Slider ───────────────────────────────────────────── */

interface RangeSliderProps {
  value?: [number, number]
  defaultValue?: [number, number]
  min?: number
  max?: number
  step?: number
  onValueChange?: (value: [number, number]) => void
  label?: string
  /** Show current values even when no label is present */
  showValue?: boolean
  formatValue?: (value: number) => string
  className?: string
}

function RangeSlider({
  value: controlledValue,
  defaultValue = [0, 100],
  min = 0,
  max = 100,
  step = 1,
  onValueChange,
  label,
  showValue = false,
  formatValue,
  className,
}: RangeSliderProps) {
  const [internalValue, setInternalValue] = useState<[number, number]>(defaultValue)
  const isControlled = controlledValue !== undefined
  const [lo, hi] = isControlled ? controlledValue! : internalValue

  const update = (next: [number, number]) => {
    if (!isControlled) setInternalValue(next)
    onValueChange?.(next)
  }

  const fmt = (v: number) => (formatValue ? formatValue(v) : String(v))

  return (
    <div className={cn('space-y-1.5', className)}>
      {(label || showValue) && (
        <div className={cn('flex justify-between', !label && 'justify-end')}>
          {label && <span className="text-xs font-medium text-muted-foreground">{label}</span>}
          {(label || showValue) && (
            <span className="text-xs font-semibold text-foreground tabular-nums">
              {fmt(lo)} – {fmt(hi)}
            </span>
          )}
        </div>
      )}
      <div className="relative flex items-center h-5">
        {/* Track */}
        <div className="absolute inset-x-0 h-2 bg-muted rounded-full" />
        {/* Filled range */}
        <div
          className="absolute h-2 bg-primary rounded-full pointer-events-none"
          style={{
            left: `${((lo - min) / (max - min)) * 100}%`,
            right: `${100 - ((hi - min) / (max - min)) * 100}%`,
          }}
        />
        {/* Low handle */}
        <input
          type="range"
          min={min}
          max={hi - step}
          step={step}
          value={lo}
          onChange={(e) => update([Number(e.target.value), hi])}
          className="absolute w-full appearance-none bg-transparent cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:shadow-md focus-visible:outline-none"
        />
        {/* High handle */}
        <input
          type="range"
          min={lo + step}
          max={max}
          step={step}
          value={hi}
          onChange={(e) => update([lo, Number(e.target.value)])}
          className="absolute w-full appearance-none bg-transparent cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:shadow-md focus-visible:outline-none"
        />
      </div>
    </div>
  )
}

export { Slider, RangeSlider }
