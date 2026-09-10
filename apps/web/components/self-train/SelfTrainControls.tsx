import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, cn } from '@sloughgpt/strui'

interface SelfTrainControlsProps {
  model: string
  temperature: number
  forever: boolean
  isRunning: boolean
  starting?: boolean
  onModelChange: (value: string) => void
  onTemperatureChange: (value: number) => void
  onForeverToggle: () => void
  onStart: () => void
  onStop: () => void
}

export function SelfTrainControls({
  model,
  temperature,
  forever,
  isRunning,
  starting,
  onModelChange,
  onTemperatureChange,
  onForeverToggle,
  onStart,
  onStop,
}: SelfTrainControlsProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Controls</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="flex flex-col gap-1">
            <Label htmlFor="st-model" variant="uppercase">Model (optional)</Label>
            <Input id="st-model" value={model} onChange={e => onModelChange(e.target.value)}
              placeholder="gpt2" className="h-8 text-xs font-mono" disabled={isRunning} />
          </div>
          <div className="flex flex-col gap-1">
            <Label htmlFor="st-temp" variant="uppercase">Temperature</Label>
            <Input id="st-temp" type="number" min={0} max={2} step={0.1} value={temperature}
              onChange={e => onTemperatureChange(Number(e.target.value))}
              className="h-8 text-xs font-mono" disabled={isRunning} />
          </div>
          <div className="flex flex-col gap-1">
            <Label variant="uppercase">Mode</Label>
            <button
              data-testid="mode-toggle"
              type="button"
              onClick={onForeverToggle}
              aria-pressed={forever}
              aria-label={forever ? 'Switch to single pass mode' : 'Switch to train forever mode'}
              disabled={isRunning}
              className={cn('h-8 rounded border px-2 text-xs transition-colors', forever ? 'border-primary bg-primary/10 text-primary' : 'border-border')}
            >
              {forever ? 'Train forever' : 'Single pass'}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isRunning ? (
            <Button variant="destructive" size="sm" onClick={onStop}>Stop</Button>
          ) : (
            <Button size="sm" onClick={onStart} disabled={starting}>
              {starting ? 'Starting...' : 'Start self-training'}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
