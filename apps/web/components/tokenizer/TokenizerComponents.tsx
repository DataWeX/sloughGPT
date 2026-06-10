'use client'

import { useState, useEffect } from 'react'
import { tokenizerController } from '@/lib/tokenizer-controller'

export function TokenBadge({ token, isSpecial }: { token: string; isSpecial?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono leading-tight border ${
      isSpecial
        ? 'bg-primary/10 border-primary/30 text-primary'
        : 'bg-card border-border/60 text-foreground'
    }`}>
      {token === ' ' ? <span className="text-warning">␣</span> : token === '\n' ? <span className="text-warning">↵</span> : token}
    </span>
  )
}

export function SegBar({ pct, label }: { pct: number; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-8 text-right text-muted-foreground tabular-nums">{pct.toFixed(0)}%</span>
      <div className="flex-1 h-4 bg-muted rounded-sm overflow-hidden">
        <div className="h-full bg-primary/40 rounded-sm" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[10px] truncate max-w-[200px] text-foreground">{label}</span>
    </div>
  )
}

export function SampleTable() {
  const [samples, setSamples] = useState<{ word: string; ids: number[]; tokens: string[]; count: number }[]>([])
  useEffect(() => {
    tokenizerController.getSamples().then(r => setSamples(r.samples || [])).catch(() => {})
  }, [])
  if (samples.length === 0) {
    return <p className="text-xs text-muted-foreground py-4 text-center">No samples available — tokenizer may not be trained</p>
  }
  return (
    <div>
      <div className="flex items-center gap-3 py-1.5 text-[10px] font-medium text-muted-foreground border-b border-border/40 uppercase tracking-wider">
        <span className="w-20">Word</span>
        <span className="w-6 text-center">#</span>
        <span>Tokens</span>
        <span className="ml-auto">IDs</span>
      </div>
      <div className="divide-y divide-border/10">
        {samples.map((s, i) => (
          <div key={i} className="flex items-center gap-3 py-1.5 text-sm border-b border-border/20 last:border-0">
            <span className="w-20 font-mono text-xs text-muted-foreground truncate shrink-0">{s.word}</span>
            <span className="w-6 text-[10px] text-muted-foreground text-center shrink-0">{s.count}</span>
            <div className="flex flex-wrap gap-1">
              {s.tokens.map((t, j) => (
                <TokenBadge key={j} token={t} />
              ))}
            </div>
            <span className="text-[10px] text-muted-foreground ml-auto font-mono">
              [{s.ids.join(', ')}]
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
