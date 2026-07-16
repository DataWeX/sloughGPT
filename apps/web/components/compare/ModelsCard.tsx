'use client'

import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { IconCheck, IconTrash } from '@sloughgpt/strui'
import type { BenchmarkResult } from '@/lib/benchmark-controller'

interface ModelEntry {
  id: string
  name: string
  loaded: boolean
  sizeGb?: number
}

interface ModelsCardProps {
  models: ModelEntry[]
  loading: boolean
  results: Record<string, BenchmarkResult | null>
  running: Set<string>
  onBenchmark: (modelId: string) => void
  onClear: (modelId: string) => void
}

export default function ModelsCard({ models, loading, results, running, onBenchmark, onClear }: ModelsCardProps) {
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Models</CardTitle></CardHeader>
      <CardContent>
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-28 rounded-lg" />)}
          </div>
        ) : models.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No models available. Load one in the Models page first.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {models.map(m => {
              const result = results[m.id]
              const isRunning = running.has(m.id)
              return (
                <div key={m.id} className={cn("rounded-lg border p-3 transition-all", result && !result.error ? "border-primary/30 bg-primary/5" : m.loaded ? "border-border/60 bg-card/50" : "border-border/30 bg-muted/20 opacity-70")}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-medium truncate">{m.name}</p>
                    {m.loaded && <Badge label="Loaded" variant="success" size="sm" />}
                  </div>
                  {m.sizeGb && <p className="text-xs text-muted-foreground mb-2">{m.sizeGb.toFixed(1)} GB</p>}
                  <div className="flex items-center gap-1.5">
                    <Button size="sm" variant={result ? "outline" : "default"} className="h-7 text-xs flex-1" onClick={() => onBenchmark(m.id)} disabled={isRunning}>
                      {isRunning ? 'Benchmarking…' : result ? <><IconCheck className="h-3 w-3 mr-1" /> Rerun</> : 'Benchmark'}
                    </Button>
                    {result && (
                      <Button variant="ghost" size="icon-sm" className="h-7 w-7" onClick={() => onClear(m.id)} aria-label={`Clear result for ${m.id}`}>
                        <IconTrash className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                  {result?.error && <p className="text-[10px] text-destructive mt-1">{result.error}</p>}
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
