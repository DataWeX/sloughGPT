'use client'

import { forwardRef, type InputHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange' | 'value' | 'defaultValue'> {
  value?: number[]
  defaultValue?: number[]
  min?: number
  max?: number
  step?: number
  onValueChange?: (value: number[]) => void
}

const Slider = forwardRef<HTMLInputElement, SliderProps>(
  ({ className, value, defaultValue, min = 0, max = 100, step = 1, onValueChange, disabled, ...props }, ref) => {
    const currentValue = value?.[0] ?? defaultValue?.[0] ?? 0

    return (
      <input
        ref={ref}
        type="range"
        min={min}
        max={max}
        step={step}
        value={currentValue}
        disabled={disabled}
        onChange={(e) => onValueChange?.([Number(e.target.value)])}
        className={cn(
          'w-full h-2 bg-muted rounded-full appearance-none cursor-pointer',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:shadow-sm',
          '[&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-primary [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-background [&::-moz-range-thumb]:shadow-sm',
          className,
        )}
        {...props}
      />
    )
  },
)
Slider.displayName = 'Slider'

interface RangeSliderProps {
  value?: [number, number]
  min?: number
  max?: number
  step?: number
  onValueChange?: (value: [number, number]) => void
  className?: string
}

function RangeSlider({ value = [0, 100], min = 0, max = 100, step = 1, onValueChange, className }: RangeSliderProps) {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <span className="text-xs text-muted-foreground">{value[0]}</span>
      <div className="relative flex-1">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[0]}
          onChange={(e) => onValueChange?.([Number(e.target.value), value[1]])}
          className="absolute w-full"
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[1]}
          onChange={(e) => onValueChange?.([value[0], Number(e.target.value)])}
          className="relative w-full"
        />
      </div>
      <span className="text-xs text-muted-foreground">{value[1]}</span>
    </div>
  )
}

export { Slider, RangeSlider }
