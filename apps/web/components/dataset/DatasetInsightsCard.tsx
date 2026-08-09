'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import type { DatasetPreview } from '@/lib/dataset-controller'

interface DatasetInsightsCardProps {
  preview: DatasetPreview | null
  loading?: boolean
}

function computeInsights(preview: DatasetPreview) {
  const { samples, languages, total_chars: totalChars } = preview
  if (samples.length === 0) return null

  const wordCounts = samples.map(s => s.content.split(/\s+/).filter(Boolean).length)
  const lineLengths = samples.map(s => s.content.length)
  const avgWords = wordCounts.reduce((a, b) => a + b, 0) / wordCounts.length
  const avgLen = lineLengths.reduce((a, b) => a + b, 0) / lineLengths.length
  const maxWords = Math.max(...wordCounts)
  const minWords = Math.min(...wordCounts)
  const emptyLines = samples.filter(s => s.content.trim().length === 0).length
  const shortLines = samples.filter(s => s.content.trim().length > 0 && s.content.trim().length < 20).length
  const longLines = samples.filter(s => s.content.length > 1000).length
  const uniqueSizes = new Set(samples.map(s => s.size))

  const langEntries = Object.entries(languages).sort((a, b) => b[1] - a[1])
  const topLang = langEntries[0]

  return {
    avgWords: Math.round(avgWords),
    avgLen: Math.round(avgLen),
    maxWords,
    minWords,
    emptyLines,
    shortLines,
    longLines,
    totalSamples: samples.length,
    totalChars,
    langEntries,
    topLang: topLang ? topLang[0] : 'unknown',
    uniqueSizes: uniqueSizes.size,
    diversityScore: Math.min(100, Math.round((uniqueSizes.size / samples.length) * 100)),
  }
}

export function DatasetInsightsCard({ preview, loading }: DatasetInsightsCardProps) {
  if (loading) {
    return (
      <Card data-testid="dataset-insights">
        <CardHeader><CardTitle className="text-base">Insights</CardTitle></CardHeader>
        <CardContent><div className="h-20 animate-pulse bg-muted/50 rounded" /></CardContent>
      </Card>
    )
  }

  if (!preview || preview.samples.length === 0) return null

  const insights = computeInsights(preview)
  if (!insights) return null

  return (
    <Card data-testid="dataset-insights">
      <CardHeader>
        <CardTitle className="text-base">Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Avg Words</div>
            <div className="text-sm font-mono font-medium">{insights.avgWords}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Avg Length</div>
            <div className="text-sm font-mono font-medium">{insights.avgLen} chars</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Diversity</div>
            <div className={`text-sm font-mono font-medium ${insights.diversityScore > 80 ? 'text-success' : insights.diversityScore > 50 ? 'text-warning' : 'text-destructive'}`}>
              {insights.diversityScore}%
            </div>
          </div>
        </div>

        {insights.langEntries.length > 0 && (
          <div className="mb-3">
            <div className="text-[10px] text-muted-foreground mb-1">Languages</div>
            <div className="flex flex-wrap gap-1">
              {insights.langEntries.slice(0, 5).map(([lang, count]) => (
                <span key={lang} className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                  {lang} ({count})
                </span>
              ))}
              {insights.langEntries.length > 5 && (
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                  +{insights.langEntries.length - 5} more
                </span>
              )}
            </div>
          </div>
        )}

        <div className="space-y-1 text-[11px]">
          {insights.emptyLines > 0 && (
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Empty samples</span>
              <span className="font-mono text-warning">{insights.emptyLines}</span>
            </div>
          )}
          {insights.shortLines > 0 && (
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Very short (&lt;20 chars)</span>
              <span className="font-mono text-warning">{insights.shortLines}</span>
            </div>
          )}
          {insights.longLines > 0 && (
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Very long (&gt;1K chars)</span>
              <span className="font-mono">{insights.longLines}</span>
            </div>
          )}
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Word range</span>
            <span className="font-mono">{insights.minWords}–{insights.maxWords}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
