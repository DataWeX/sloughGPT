'use client'

import { Card, CardContent, CardHeader, CardTitle, cn } from '@sloughgpt/strui'

interface EvalQualityCardProps {
  coherenceScore?: number
  qualityScore?: number
  repetitionRate?: number
}

export function EvalQualityCard({
  coherenceScore,
  qualityScore,
  repetitionRate,
}: EvalQualityCardProps) {
  const hasData = coherenceScore != null || qualityScore != null || repetitionRate != null

  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Quality Metrics</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        {hasData ? (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5">
            {[
              { label: 'Coherence', value: `${((coherenceScore ?? 0) * 100).toFixed(1)}%`, color: 'text-success' },
              { label: 'Quality', value: `${((qualityScore ?? 0) * 100).toFixed(1)}%`, color: 'text-primary' },
              { label: 'Repetition', value: `${((repetitionRate ?? 0) * 100).toFixed(1)}%`, color: (repetitionRate ?? 0) > 0.3 ? 'text-destructive' : 'text-muted-foreground' },
            ].map(s => (
              <div key={s.label} className="rounded-lg bg-muted/20 p-2.5 text-center">
                <div className="text-[10px] text-muted-foreground">{s.label}</div>
                <div className={cn('text-[11px] font-mono font-medium', s.color)}>{s.value}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-6 text-[10px] text-muted-foreground/60">
            No quality data yet. Chat with the model to generate responses.
          </div>
        )}
      </CardContent>
    </Card>
  )
}
