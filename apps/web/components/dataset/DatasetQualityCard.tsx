'use client'

import { useState, useEffect, useMemo, useCallback, memo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, StatCard, KpiGrid } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { datasetController, type DatasetPreview } from '@/lib/dataset-controller'

interface DatasetQualityCardProps {
  datasetId: string
}

interface QualityMetrics {
  totalLines: number
  emptyLines: number
  duplicateLines: number
  avgLineLength: number
  medianLineLength: number
  shortLines: number
  longLines: number
  nonAsciiLines: number
  duplicateRatio: number
  emptyRatio: number
}

function computeQuality(preview: DatasetPreview): QualityMetrics {
  const allLines = preview.samples.map(s => s.content)
  const nonEmptyLines = allLines.filter(l => l && l.trim().length > 0)
  const totalLines = preview.total_samples || allLines.length
  const lengths = nonEmptyLines.map(l => l.length).sort((a, b) => a - b)
  const uniqueLines = new Set(nonEmptyLines)

  return {
    totalLines,
    emptyLines: allLines.filter(l => !l || l.trim().length === 0).length,
    duplicateLines: nonEmptyLines.length - uniqueLines.size,
    avgLineLength: lengths.length > 0 ? lengths.reduce((a, b) => a + b, 0) / lengths.length : 0,
    medianLineLength: lengths.length > 0 ? lengths[Math.floor(lengths.length / 2)] : 0,
    shortLines: nonEmptyLines.filter(l => l.length < 10).length,
    longLines: nonEmptyLines.filter(l => l.length > 1000).length,
    nonAsciiLines: nonEmptyLines.filter(l => /[^\x00-\x7F]/.test(l)).length,
    duplicateRatio: nonEmptyLines.length > 0 ? (nonEmptyLines.length - uniqueLines.size) / nonEmptyLines.length : 0,
    emptyRatio: totalLines > 0 ? allLines.filter(l => !l || l.trim().length === 0).length / totalLines : 0,
  }
}

export const DatasetQualityCard = memo(function DatasetQualityCard({ datasetId }: DatasetQualityCardProps) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchPreview = useCallback(async () => {
    setLoading(true)
    try {
      const p = await datasetController.preview(datasetId, 200)
      setPreview(p)
    } catch {
      setPreview(null)
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const p = await datasetController.preview(datasetId, 200)
        if (active) setPreview(p)
      } catch {
        if (active) setPreview(null)
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [datasetId])

  const metrics = useMemo(() => preview ? computeQuality(preview) : null, [preview])

  if (!preview || !metrics) {
    return (
      <Card data-testid="dataset-quality">
        <CardHeader>
          <CardTitle className="text-base">Data Quality</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground text-center py-4">
            {loading ? 'Analyzing data quality...' : 'Preview data to analyze quality.'}
          </p>
        </CardContent>
      </Card>
    )
  }

  const issues: string[] = []
  if (metrics.emptyRatio > 0.05) issues.push(`${(metrics.emptyRatio * 100).toFixed(0)}% empty lines`)
  if (metrics.duplicateRatio > 0.1) issues.push(`${(metrics.duplicateRatio * 100).toFixed(0)}% duplicates`)
  if (metrics.nonAsciiLines > metrics.totalLines * 0.3) issues.push('High non-ASCII content')
  if (metrics.shortLines > metrics.totalLines * 0.2) issues.push('Many very short lines')
  if (metrics.longLines > 0) issues.push(`${metrics.longLines} very long lines`)

  return (
    <Card data-testid="dataset-quality">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">Data Quality</CardTitle>
            {issues.length === 0 && (
              <Badge label="Good" variant="success" size="sm" />
            )}
            {issues.length > 0 && issues.length <= 2 && (
              <Badge label="Fair" variant="warning" size="sm" />
            )}
            {issues.length > 2 && (
              <Badge label="Poor" variant="error" size="sm" />
            )}
          </div>
          <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={fetchPreview} disabled={loading} aria-label="Refresh preview">
            <IconRefresh className={loading ? 'animate-spin h-3 w-3' : 'h-3 w-3'} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <KpiGrid columns={3}>
            <StatCard label="Total lines" value={metrics.totalLines.toLocaleString()} />
            <StatCard label="Avg chars" value={metrics.avgLineLength.toFixed(0)} />
            <StatCard label="Median chars" value={metrics.medianLineLength} />
          </KpiGrid>

          <div className="space-y-1.5">
            {metrics.emptyLines > 0 && (
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-muted-foreground">Empty lines</span>
                <span className="font-mono">{metrics.emptyLines} ({(metrics.emptyRatio * 100).toFixed(1)}%)</span>
              </div>
            )}
            {metrics.duplicateLines > 0 && (
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-muted-foreground">Duplicates</span>
                <span className="font-mono">{metrics.duplicateLines} ({(metrics.duplicateRatio * 100).toFixed(1)}%)</span>
              </div>
            )}
            {metrics.shortLines > 0 && (
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-muted-foreground">Short lines (&lt;10 chars)</span>
                <span className="font-mono">{metrics.shortLines}</span>
              </div>
            )}
            {metrics.longLines > 0 && (
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-muted-foreground">Long lines (&gt;1K chars)</span>
                <span className="font-mono">{metrics.longLines}</span>
              </div>
            )}
            {metrics.nonAsciiLines > 0 && (
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-muted-foreground">Non-ASCII</span>
                <span className="font-mono">{metrics.nonAsciiLines}</span>
              </div>
            )}
          </div>

          {issues.length > 0 && (
            <div className="pt-2 border-t border-border/40">
              <p className="text-[10px] text-muted-foreground mb-1">Issues</p>
              <div className="flex flex-wrap gap-1">
                {issues.map((issue, i) => (
                   <Badge key={i} label={issue} variant="outline" size="sm" />
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
})
