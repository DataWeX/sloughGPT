'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface WeightBar {
  label: string
  value: number
  max: number
}

interface WeightsTestCardProps {
  testMessage: string
  onMessageChange: (m: string) => void
  onCompute: () => void
  testing?: boolean
  weights?: {
    temperature: number
    top_p: number
    repetition_penalty: number
    style_bias: number
    confidence_boost: number
    top_k: number
    based_on_samples: number
  } | null
}

export function WeightsTestCard({ testMessage, onMessageChange, onCompute, testing, weights }: WeightsTestCardProps) {
  const weightBars: WeightBar[] = weights ? [
    { label: 'Temperature', value: weights.temperature, max: 2.0 },
    { label: 'Top P', value: weights.top_p, max: 1.0 },
    { label: 'Repetition Penalty', value: weights.repetition_penalty, max: 2.0 },
    { label: 'Style Bias', value: weights.style_bias, max: 1.0 },
    { label: 'Confidence Boost', value: weights.confidence_boost, max: 1.0 },
  ] : []

  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Test Weights</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <p className="text-[10px] text-muted-foreground/60">Enter a message to see what meta-weight adjustments the system would apply based on similar past feedback.</p>
        <div className="flex gap-1.5">
          <Input
            value={testMessage}
            onChange={e => onMessageChange(e.target.value)}
            placeholder="Type a message to test..."
            className="h-7 text-[11px] flex-1"
            onKeyDown={e => e.key === 'Enter' && onCompute()}
          />
          <Button size="sm" className="h-7 text-[11px]" onClick={onCompute} disabled={testing || !testMessage.trim()}>
            {testing ? 'Computing...' : 'Compute'}
          </Button>
        </div>

        {weights && (
          <div className="space-y-2 pt-1">
            <p className="text-[10px] text-muted-foreground/60">Based on {weights.based_on_samples} feedback samples</p>

            <div className="space-y-1.5">
              {weightBars.map(w => (
                <div key={w.label} className="space-y-0.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-medium">{w.label}</span>
                    <span className="text-[10px] font-mono tabular-nums text-muted-foreground/60">{w.value.toFixed(3)}</span>
                  </div>
                  <div className="h-1 bg-muted/40 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary/60 rounded-full transition-all"
                      style={{ width: `${Math.min(100, (w.value / w.max) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-1.5 pt-1">
              <div className="border border-border/40 rounded-lg p-2">
                <span className="text-[9px] text-muted-foreground/60 uppercase tracking-wider font-medium">Top K</span>
                <p className="text-[11px] font-semibold tabular-nums mt-0.5">{weights.top_k}</p>
              </div>
              <div className="border border-border/40 rounded-lg p-2">
                <span className="text-[9px] text-muted-foreground/60 uppercase tracking-wider font-medium">Based on Samples</span>
                <p className="text-[11px] font-semibold tabular-nums mt-0.5">{weights.based_on_samples}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
