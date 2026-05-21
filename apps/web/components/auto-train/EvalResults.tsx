'use client'

interface EvalData {
  baseline: Record<string, number | string>
  with_adapter: Record<string, number | string>
  delta: Record<string, number | string>
  report?: string
}

interface EvalResultsProps {
  data: EvalData
}

function MetricRow({
  label,
  baseline,
  adapter,
  delta,
  colorGreenWhen,
}: {
  label: string
  baseline: number | undefined
  adapter: number | undefined
  delta: number | undefined
  colorGreenWhen: 'lower' | 'higher'
}) {
  if (baseline === undefined) return null

  const fmt = (v: number, decimals = 2) => {
    if (Math.abs(v) < 0.01) return v.toFixed(4)
    return v.toFixed(decimals)
  }
  const deltaVal = delta ?? 0
  const positive = colorGreenWhen === 'higher' ? deltaVal > 0 : deltaVal < 0

  return (
    <div className="space-y-1 p-2 rounded bg-muted/20">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-sm font-mono">
        {fmt(baseline)} → {adapter !== undefined ? fmt(adapter) : '--'}
        {delta !== undefined && (
          <span className={positive ? 'text-success' : 'text-destructive'}>
            {' '}({deltaVal > 0 ? '+' : ''}{fmt(deltaVal)})
          </span>
        )}
      </div>
    </div>
  )
}

function EvalResults({ data }: EvalResultsProps) {
  const { baseline, with_adapter, delta } = data

  const verdict = delta?.verdict as string | undefined
  const verdictColor =
    verdict === 'improved' ? 'bg-success/10 text-success' :
    verdict === 'degraded' ? 'bg-destructive/10 text-destructive' :
    'bg-warning/10 text-warning'

  return (
    <div className="space-y-3">
      {verdict && (
        <div className={`text-sm font-bold text-center py-1.5 rounded ${verdictColor}`}>
          Verdict: {verdict.toUpperCase()}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2">
        <MetricRow
          label="Perplexity"
          baseline={baseline.perplexity as number}
          adapter={with_adapter.perplexity as number}
          delta={delta.perplexity_delta as number}
          colorGreenWhen="lower"
        />
        <MetricRow
          label="BLEU"
          baseline={baseline.bleu as number}
          adapter={with_adapter.bleu as number}
          delta={delta.bleu_delta as number}
          colorGreenWhen="higher"
        />
        <MetricRow
          label="Throughput"
          baseline={baseline.tokens_per_sec as number}
          adapter={with_adapter.tokens_per_sec as number}
          delta={delta.throughput_delta as number}
          colorGreenWhen="higher"
        />
        <MetricRow
          label="Personality Score"
          baseline={baseline.personality_score as number}
          adapter={with_adapter.personality_score as number}
          delta={delta.personality_delta as number}
          colorGreenWhen="higher"
        />
      </div>
      {data.report && (
        <details className="text-xs text-muted-foreground border border-border rounded p-2">
          <summary className="cursor-pointer font-medium">View Report</summary>
          <pre className="mt-2 whitespace-pre-wrap font-mono text-xs">{data.report}</pre>
        </details>
      )}
    </div>
  )
}

export { EvalResults }
export type { EvalData }
