'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { Slider } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'

export interface GenerationConfigValues {
  temperature: number
  top_p: number
  top_k: number
  max_new_tokens: number
  repetition_penalty: number
}

interface GenerationConfigCardProps {
  values: GenerationConfigValues
  onChange: (values: Partial<GenerationConfigValues>) => void
  onReset?: () => void
  showRepetitionPenalty?: boolean
  compact?: boolean
}

const DEFAULTS: GenerationConfigValues = {
  temperature: 0.7,
  top_p: 0.85,
  top_k: 40,
  max_new_tokens: 300,
  repetition_penalty: 1.15,
}

function SliderRow({ label, value, onChange, min, max, step, format }: {
  label: string; value: number; onChange: (v: number) => void
  min: number; max: number; step: number; format?: (v: number) => string
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <span className="text-xs font-mono text-foreground">{format ? format(value) : value}</span>
      </div>
      <Slider value={[value]} onValueChange={([v]: number[]) => onChange(v)} min={min} max={max} step={step} />
    </div>
  )
}

export function GenerationConfigCard({ values, onChange, onReset, showRepetitionPenalty = true, compact = false }: GenerationConfigCardProps) {
  const isDefault = Object.keys(DEFAULTS).every(k => {
    const key = k as keyof GenerationConfigValues
    if (key === 'repetition_penalty' && !showRepetitionPenalty) return true
    return Math.abs((values[key] ?? 0) - DEFAULTS[key]) < 0.01
  })

  return (
    <Card data-testid="generation-config">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Generation Config</CardTitle>
          {!isDefault && onReset && (
            <Button size="sm" variant="ghost" className="h-7 text-xs text-muted-foreground" onClick={onReset}>
              Reset
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className={compact ? 'space-y-3' : 'grid grid-cols-1 sm:grid-cols-2 gap-4'}>
          <SliderRow label="Temperature" value={values.temperature} onChange={v => onChange({ temperature: v })} min={0} max={2} step={0.05} format={v => v.toFixed(2)} />
          <SliderRow label="Top-P" value={values.top_p} onChange={v => onChange({ top_p: v })} min={0} max={1} step={0.05} format={v => v.toFixed(2)} />
          <SliderRow label="Top-K" value={values.top_k} onChange={v => onChange({ top_k: v })} min={0} max={100} step={1} />
          <SliderRow label="Max Tokens" value={values.max_new_tokens} onChange={v => onChange({ max_new_tokens: v })} min={1} max={2048} step={1} />
          {showRepetitionPenalty && (
            <SliderRow label="Repetition Penalty" value={values.repetition_penalty} onChange={v => onChange({ repetition_penalty: v })} min={1} max={2} step={0.05} format={v => v.toFixed(2)} />
          )}
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
            temp={values.temperature.toFixed(2)}
          </span>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
            p={values.top_p.toFixed(2)}
          </span>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
            k={values.top_k}
          </span>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
            tokens={values.max_new_tokens}
          </span>
        </div>
      </CardContent>
    </Card>
  )
}

export { DEFAULTS as GENERATION_DEFAULTS }
