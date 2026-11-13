'use client'

/**
 * Shared UI atoms for the Settings page: the version badge and the
 * labelled slider. Pure render.
 */

import { Badge, Slider } from '@sloughgpt/strui'
export const WEB_VERSION = '3.0.0'

export function VersionBadge({ label, version }: { label: string; version?: string | null }) {
  if (!version) return null
  return (
    <Badge
      variant="outline"
      className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border-border/40"
    >
      {label} {version}
    </Badge>
  )
}

export function SettingsSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  formatValue,
}: {
  label: string
  value: number | undefined
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  formatValue?: (v: number) => string
}) {
  const safeValue = value ?? 0
  const display = formatValue ? formatValue(safeValue) : `${safeValue}`
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <span className="text-sm text-muted-foreground">{display}</span>
      </div>
      <Slider
        value={[safeValue]}
        onValueChange={([v]: number[]) => onChange(v)}
        min={min}
        max={max}
        step={step}
      />
    </div>
  )
}
