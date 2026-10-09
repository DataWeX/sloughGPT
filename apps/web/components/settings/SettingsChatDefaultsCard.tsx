'use client'

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Button, Slider, Switch } from '@sloughgpt/strui'

interface SettingsChatDefaultsCardProps {
  temperature: number
  maxTokens: number
  topP: number
  topK: number
  streaming: boolean
  collapsibleMessageLength: number
  onTemperatureChange: (v: number) => void
  onMaxTokensChange: (v: number) => void
  onTopPChange: (v: number) => void
  onTopKChange: (v: number) => void
  onStreamingChange: (v: boolean) => void
  onCollapsibleMessageLengthChange: (v: number) => void
  onReset: () => void
  version?: string
}

function SettingsSlider({
  label, value, onChange, min, max, step, formatValue,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  formatValue?: (v: number) => string
}) {
  const display = formatValue ? formatValue(value) : `${value}`
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <span className="text-sm text-muted-foreground">{display}</span>
      </div>
      <Slider value={[value]} onValueChange={([v]: number[]) => onChange(v)} min={min} max={max} step={step} />
    </div>
  )
}

export function SettingsChatDefaultsCard({
  temperature,
  maxTokens,
  topP,
  topK,
  streaming,
  collapsibleMessageLength,
  onTemperatureChange,
  onMaxTokensChange,
  onTopPChange,
  onTopKChange,
  onStreamingChange,
  onCollapsibleMessageLengthChange,
  onReset,
  version,
}: SettingsChatDefaultsCardProps) {
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="text-base">Chat defaults</CardTitle>
          <CardDescription>Default model and generation settings</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <SettingsSlider label="Temperature" value={temperature} onChange={onTemperatureChange} min={0} max={2} step={0.1} />
          <SettingsSlider label="Max tokens" value={maxTokens} onChange={onMaxTokensChange} min={50} max={1000} step={50} />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <SettingsSlider label="Top-P" value={topP} onChange={onTopPChange} min={0} max={1} step={0.05} />
          <SettingsSlider label="Top-K" value={topK} onChange={onTopKChange} min={0} max={100} step={5} />
        </div>
        <div className="flex items-center justify-between pt-2">
          <div>
            <p className="text-sm font-medium">Streaming</p>
            <p className="text-xs text-muted-foreground">Show tokens as they are generated</p>
          </div>
          <Switch checked={streaming} onCheckedChange={onStreamingChange} aria-label="Toggle streaming" />
        </div>
        <div className="pt-2">
          <SettingsSlider
            label="Auto-collapse messages longer than"
            value={collapsibleMessageLength}
            onChange={onCollapsibleMessageLengthChange}
            min={0}
            max={2000}
            step={50}
            formatValue={(v) => v === 0 ? 'Disabled' : `${v} chars`}
          />
        </div>
        <div className="flex items-center justify-between pt-3 border-t border-border/40">
          <p className="text-xs text-muted-foreground">Reset generation defaults</p>
          <Button size="sm" variant="ghost" className="h-8 text-xs text-muted-foreground" onClick={onReset}>Reset</Button>
        </div>
      </CardContent>
      <CardFooter className="justify-end">
        {version && <span className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border border-border/50 rounded text-xs">v{version}</span>}
      </CardFooter>
    </Card>
  )
}
