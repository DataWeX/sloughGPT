interface LossCurveProps {
  data: Array<{ step: number; loss: number }>
}

export function LossCurve({ data }: LossCurveProps) {
  const W = 220, H = 72, P = 8
  const losses = data.map(d => d.loss)
  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const range = max - min || 1
  const steps = data.map(d => d.step)
  const stepMin = Math.min(...steps)
  const stepRange = Math.max(...steps) - stepMin || 1

  const toX = (s: number) => P + ((s - stepMin) / stepRange) * (W - 2 * P)
  const toY = (l: number) => P + ((max - l) / range) * (H - 2 * P)

  const points = data.map(d => `${toX(d.step)},${toY(d.loss)}`).join(' ')
  const fillPath = data.length >= 2
    ? `M${data.map(d => `${toX(d.step)},${toY(d.loss)}`).join(' L')} L${toX(data[data.length - 1].step)},${H - P} L${toX(data[0].step)},${H - P} Z`
    : ''

  return (
    <div className="p-2 rounded bg-muted/30 border border-border/40">
      <div className="text-[10px] text-muted-foreground mb-1">Loss curve</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" aria-label="Training loss over steps">
        <defs>
          <linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.2" />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {data.length >= 2 && (
          <>
            <path d={fillPath} fill="url(#loss-fill)" />
            <polyline
              points={points}
              fill="none"
              stroke="hsl(var(--primary))"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <circle
              cx={toX(data[data.length - 1].step)}
              cy={toY(data[data.length - 1].loss)}
              r="2"
              fill="hsl(var(--primary))"
            />
          </>
        )}
      </svg>
      <div className="flex justify-between text-[9px] text-muted-foreground mt-0.5">
        <span>step {stepMin}</span>
        <span>{max.toFixed(2)} &rarr; {min.toFixed(2)}</span>
        <span>step {Math.max(...steps)}</span>
      </div>
    </div>
  )
}
