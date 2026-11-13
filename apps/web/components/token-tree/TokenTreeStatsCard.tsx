'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

export interface TokenTreeStats {
  trained: boolean
  vocab_size: number
  num_merges: number
  embedding_points: number
  num_base_tokens: number
  embedding_compression_ratio: number
  embed_dim: number
}

export interface TokenTreeStatsCardProps {
  stats: TokenTreeStats
}

export function TokenTreeStatsCard({ stats }: TokenTreeStatsCardProps) {
  const items = [
    { label: 'Trained', value: stats.trained ? 'Yes' : 'No' },
    { label: 'Vocab Size', value: stats.vocab_size.toLocaleString() },
    { label: 'Merges', value: stats.num_merges.toLocaleString() },
    { label: 'Embeddings', value: stats.embedding_points.toLocaleString() },
    { label: 'Base Tokens', value: stats.num_base_tokens.toLocaleString() },
    { label: 'Compression', value: stats.embedding_compression_ratio.toFixed(2) },
    { label: 'Embed Dim', value: stats.embed_dim },
  ]

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Token Tree Stats</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
          {items.map(s => (
            <div key={s.label} className="rounded-lg bg-muted/20 p-2 text-center">
              <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">{s.label}</div>
              <div className="text-[11px] font-mono font-medium tabular-nums mt-0.5">{s.value}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
