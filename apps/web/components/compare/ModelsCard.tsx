'use client'

import { useState, memo } from 'react'
import { useRouter } from 'next/navigation'
import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { IconCheck, IconTrash } from '@sloughgpt/strui'
import type { BenchmarkResult } from '@/lib/benchmark-controller'
import type { ModelEntry } from '@/lib/types/models'

interface ModelsCardProps {
  models: ModelEntry[]
  loading: boolean
  results: Record<string, BenchmarkResult | null>
  running: Set<string>
  onBenchmark: (modelId: string) => void
  onClear: (modelId: string) => void
}

export default memo(function ModelsCard({ models, loading, results, running, onBenchmark, onClear }: ModelsCardProps) {
  const [search, setSearch] = useState('')
  const router = useRouter()
  const filtered = search
    ? models.filter(m => m.name.toLowerCase().includes(search.toLowerCase()) || m.id.toLowerCase().includes(search.toLowerCase()))
    : models
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs">Models</CardTitle>
          {models.length > 3 && (
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Filter..."
              aria-label="Filter models"
              className="h-6 w-28 rounded-md border border-border/40 bg-background px-1.5 text-[10px] placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
          )}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
            {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24 rounded-lg" />)}
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-5 text-[10px] text-muted-foreground/60 space-y-1.5">
            <div>No models available. Load one in the Models page first.</div>
            <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={() => router.push('/models')}>
              Open Models
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
            {filtered.map(m => {
              const result = results[m.id]
              const isRunning = running.has(m.id)
              return (
                <div key={m.id} className={cn("rounded-lg border p-2 transition-all", result && !result.error ? "border-primary/30 bg-primary/5" : m.loaded ? "border-border/40 bg-card/50" : "border-border/30 bg-muted/20 opacity-70")}>
                  <div className="flex items-center justify-between mb-1.5">
                    <p className="text-[11px] font-medium truncate">{m.name}</p>
                    {m.loaded && <Badge label="Loaded" variant="success" size="sm" />}
                  </div>
                  <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
                    {m.sizeGb && <span className="text-[9px] text-muted-foreground/60 tabular-nums">{m.sizeGb.toFixed(1)} GB</span>}
                    {m.source && (
                      <span className="text-[8px] px-1 py-0.5 rounded bg-muted text-muted-foreground/60 font-medium">{m.source}</span>
                    )}
                    {m.type && m.type !== 'text-generation' && (
                      <span className="text-[8px] px-1 py-0.5 rounded bg-primary/10 text-primary font-medium">{m.type}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="sm" variant={result ? "outline" : "default"} className="h-6 text-[10px] flex-1" onClick={() => onBenchmark(m.id)} disabled={isRunning}>
                      {isRunning ? 'Benchmarking…' : result ? <><IconCheck className="h-2.5 w-2.5 mr-0.5" /> Rerun</> : 'Benchmark'}
                    </Button>
                    {result && (
                      <Button variant="ghost" size="icon-sm" className="h-6 w-6" onClick={() => onClear(m.id)} aria-label={`Clear result for ${m.id}`}>
                        <IconTrash className="h-2.5 w-2.5" />
                      </Button>
                    )}
                  </div>
                  {result?.error && <p className="text-[9px] text-destructive mt-1">{result.error}</p>}
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
})
