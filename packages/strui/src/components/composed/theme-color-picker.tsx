'use client'

import { useState, type InputHTMLAttributes, useCallback } from 'react'
import { cn } from '../../lib/cn'

export interface ThemeSwatch {
  id: string
  name: string
  color: string
}

export const DEFAULT_THEME_SWATCHES: ThemeSwatch[] = [
  { id: 'blue', name: 'Periwinkle', color: '#8b7bc4' },
  { id: 'purple', name: 'Lilac', color: '#a67fd4' },
  { id: 'pink', name: 'Rose', color: '#d894b4' },
  { id: 'red', name: 'Coral', color: '#e88890' },
  { id: 'orange', name: 'Peach', color: '#e8a86c' },
  { id: 'green', name: 'Mint', color: '#6bb89a' },
  { id: 'teal', name: 'Dew', color: '#6cabcc' },
]

interface ThemeSwatchProps {
  swatch: ThemeSwatch
  selected: boolean
  onClick: () => void
  showLabel?: boolean
}

export function ThemeSwatch({ swatch, selected, onClick, showLabel = false }: ThemeSwatchProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'sl-hover-primary flex items-center gap-2 rounded-none transition-all duration-200',
        showLabel ? 'px-3 py-2' : 'p-1.5',
      )}
      style={{ backgroundColor: selected ? `${swatch.color}20` : undefined }}
      aria-label={`${swatch.name} theme`}
      aria-pressed={selected}
    >
      <span
        className={cn(
          'h-5 w-5 shrink-0 rounded-none transition-all duration-200',
          selected && 'ring-2 ring-offset-2 ring-offset-background scale-110 shadow-sm',
        )}
        style={{
          backgroundColor: swatch.color,
          ...(selected ? { '--tw-ring-color': swatch.color } as React.CSSProperties : {}),
        }}
      />
      {showLabel && <span className="text-sm font-medium">{swatch.name}</span>}
    </button>
  )
}

export interface ThemeColorPickerProps {
  swatches?: ThemeSwatch[]
  value: string
  onChange: (color: string) => void
  onCustomColor?: (color: string) => void
  className?: string
  showCustomInput?: boolean
  label?: string
}

export function ThemeColorPicker({
  swatches = DEFAULT_THEME_SWATCHES,
  value,
  onChange,
  onCustomColor,
  className,
  showCustomInput = true,
  label = 'Accent color',
}: ThemeColorPickerProps) {
  const [customColor, setCustomColor] = useState('')
  const [showCustom, setShowCustom] = useState(false)

  const handleCustomSubmit = useCallback(() => {
    if (customColor && onCustomColor) {
      onCustomColor(customColor)
      onChange(customColor)
    }
  }, [customColor, onCustomColor, onChange])

  return (
    <div className={cn('flex flex-col gap-3', className)}>
      {label && <span className="sl-label">{label}</span>}
      <div className="flex flex-wrap items-center gap-1" role="radiogroup" aria-label={label}>
        {swatches.map((swatch) => (
          <ThemeSwatch
            key={swatch.id}
            swatch={swatch}
            selected={value === swatch.color}
            onClick={() => onChange(swatch.color)}
          />
        ))}
        {showCustomInput && (
          <button
            type="button"
            onClick={() => setShowCustom(!showCustom)}
            className="sl-hover-primary ml-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-none border border-dashed border-border"
            aria-label="Custom color"
            title="Custom color"
          >
            <svg
              className="h-4 w-4 text-muted-foreground"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
        )}
      </div>
      {showCustom && (
        <div className="flex items-center gap-2">
          <label className="sr-only" htmlFor="custom-color">Custom color</label>
          <input
            id="custom-color"
            type="color"
            value={customColor || value}
            onChange={(e) => setCustomColor(e.target.value)}
            className="h-8 w-12 cursor-pointer rounded-none border border-border bg-transparent"
          />
          <input
            type="text"
            value={customColor || value}
            onChange={(e) => setCustomColor(e.target.value)}
            placeholder="#000000"
            className="sl-input h-8 w-24 px-2 font-mono text-xs"
          />
          {onCustomColor && (
            <button
              type="button"
              onClick={handleCustomSubmit}
              className="sl-btn-primary sl-btn h-8 px-3 text-xs"
            >
              Apply
            </button>
          )}
        </div>
      )}
    </div>
  )
}

interface ColorInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
}

export function ColorInput({ label, className, ...props }: ColorInputProps) {
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      {label && <label className="sl-label">{label}</label>}
      <div className="flex items-center gap-2">
        <input
          type="color"
          className="h-10 w-14 cursor-pointer rounded-none border border-border bg-transparent"
          {...props}
        />
        <input
          type="text"
          className="sl-input h-10 w-28 px-3 font-mono text-sm"
          {...props}
        />
      </div>
    </div>
  )
}
