'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'
import { IconRefresh, IconDownload } from '@sloughgpt/strui'
import ModelsCard from '@/components/compare/ModelsCard'
import ComparisonTableCard from '@/components/compare/ComparisonTableCard'
import SummaryCard from '@/components/compare/SummaryCard'
import { ModelComparisonInsightsCard } from '@/components/compare/ModelComparisonInsightsCard'
import dynamicNext from 'next/dynamic'
import type { ModelEntry, SavedSnapshot } from '@/hooks/useComparison'
import type { BenchmarkResult } from '@/lib/benchmark-controller'

const OutputComparisonCard = dynamicNext<{ models: ModelEntry[] }>(() => import('@/components/compare/OutputComparisonCard') as Promise<{ default: React.ComponentType<{ models: ModelEntry[] }> }>, { ssr: false })
const VisualComparisonCard = dynamicNext(() => import('@/components/compare/VisualComparisonCard'), { ssr: false })

interface ComparisonViewProps {
  models: ModelEntry[]
  loading: boolean
  results: Record<string, BenchmarkResult | null>
  running: Set<string>
  snapshots: SavedSnapshot[]
  snapshotName: string
  onSnapshotNameChange: (name: string) => void
  completedResults: [string, BenchmarkResult][]
  bestMetrics: Record<string, number>
  chartData: { name: string; throughput: number; latency: number; memory: number }[]
  onBenchmark: (modelId: string) => void
  onClear: (modelId: string) => void
  onRunAll: () => void
  onExport: () => void
  onSaveSnapshot: () => void
  onLoadSnapshot: (snap: SavedSnapshot) => void
  onDeleteSnapshot: (id: string) => void
}

export function ComparisonView({
  models,
  loading,
  results,
  running,
  snapshots,
  snapshotName,
  onSnapshotNameChange,
  completedResults,
  bestMetrics,
  chartData,
  onBenchmark,
  onClear,
  onRunAll,
  onExport,
  onSaveSnapshot,
  onLoadSnapshot,
  onDeleteSnapshot,
}: ComparisonViewProps) {
  return (
    <>
      {snapshots.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Saved Comparisons</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1.5">
              {snapshots.map(snap => (
                <div key={snap.id} className="flex items-center gap-1 rounded-lg border border-border/40 bg-muted/20 px-1.5 py-1">
                  <button type="button" onClick={() => onLoadSnapshot(snap)} className="text-[10px] font-medium hover:text-primary transition-colors">
                    {snap.name}
                  </button>
                  <span className="text-[9px] text-muted-foreground/60 tabular-nums">{new Date(snap.savedAt).toLocaleDateString()}</span>
                  <button type="button" onClick={() => onDeleteSnapshot(snap.id)} aria-label={`Delete snapshot ${snap.name}`} className="text-[10px] text-muted-foreground/60 hover:text-destructive ml-0.5">×</button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <ModelsCard models={models} loading={loading} results={results} running={running} onBenchmark={onBenchmark} onClear={onClear} />

      {completedResults.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center space-y-2">
            <p className="text-[11px] text-muted-foreground/60">No benchmark results yet.</p>
            <p className="text-[10px] text-muted-foreground/50 max-w-sm mx-auto">
              Run benchmarks on your models to see side-by-side comparisons. Click &ldquo;Benchmark all&rdquo; or use the benchmark button on each model card above.
            </p>
            <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={onRunAll} disabled={loading || models.length === 0}>
              Benchmark all
            </Button>
            <div className="flex items-center justify-center gap-3 text-[9px] text-muted-foreground/40 pt-1">
              <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">R</kbd> Benchmark all</span>
              <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">Ctrl+S</kbd> Save snapshot</span>
              <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">Ctrl+E</kbd> Export</span>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          <ComparisonTableCard completedResults={completedResults} models={models} bestMetrics={bestMetrics} />
          <ModelComparisonInsightsCard completedResults={completedResults} models={models} bestMetrics={bestMetrics} />
          <SummaryCard completedResults={completedResults} models={models} />
          <OutputComparisonCard models={models} />
          <VisualComparisonCard chartData={chartData} />
        </>
      )}
    </>
  )
}

interface ComparisonHeaderProps {
  completedResults: [string, BenchmarkResult][]
  snapshotName: string
  onSnapshotNameChange: (name: string) => void
  onSaveSnapshot: () => void
  onExport: () => void
  onRunAll: () => void
  loading: boolean
  running: Set<string>
}

export function ComparisonHeader({
  completedResults,
  snapshotName,
  onSnapshotNameChange,
  onSaveSnapshot,
  onExport,
  onRunAll,
  loading,
  running,
}: ComparisonHeaderProps) {
  return (
    <div className="flex items-center gap-1.5">
      {completedResults.length > 0 && (
        <>
          <div className="flex items-center gap-1">
            <Input
              value={snapshotName}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => onSnapshotNameChange(e.target.value)}
              placeholder="Snapshot name..."
              aria-label="Snapshot name"
              className="h-6 w-36 text-[10px]"
              onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') onSaveSnapshot() }}
            />
            <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={onSaveSnapshot}>Save</Button>
          </div>
          <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={onExport}>
            <IconDownload className="h-2.5 w-2.5 mr-0.5" />
            Export
          </Button>
        </>
      )}
      <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={onRunAll} disabled={loading || running.size > 0}>
        <IconRefresh className="h-2.5 w-2.5 mr-0.5" /> Benchmark all
      </Button>
    </div>
  )
}
