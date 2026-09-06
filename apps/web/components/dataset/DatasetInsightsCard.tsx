'use client'

import { Card, CardContent, CardHeader, CardTitle, InsightsCard, ChipGroup } from '@sloughgpt/strui'
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
    <InsightsCard
      title="Insights"
      testId="dataset-insights"
      kpis={[
        { label: 'Avg Words', value: insights.avgWords },
        { label: 'Avg Length', value: `${insights.avgLen} chars` },
        { label: 'Diversity', value: <span className={insights.diversityScore > 80 ? 'text-success' : insights.diversityScore > 50 ? 'text-warning' : 'text-destructive'}>{insights.diversityScore}%</span> },
      ]}
      kpiColumns={3}
      details={[
        ...(insights.emptyLines > 0 ? [{ label: 'Empty samples', value: <span className="font-mono text-warning">{insights.emptyLines}</span> }] : []),
        ...(insights.shortLines > 0 ? [{ label: 'Very short (<20 chars)', value: <span className="font-mono text-warning">{insights.shortLines}</span> }] : []),
        ...(insights.longLines > 0 ? [{ label: 'Very long (>1K chars)', value: <span className="font-mono">{insights.longLines}</span> }] : []),
        { label: 'Word range', value: <span className="font-mono">{insights.minWords}–{insights.maxWords}</span> },
      ]}
    >
      {insights.langEntries.length > 0 && (
        <div className="mt-2">
          <div className="text-[10px] text-muted-foreground mb-1">Languages</div>
          <ChipGroup
            chips={insights.langEntries.slice(0, 5).map(([lang, count]) => ({
              children: `${lang} (${count})`,
            }))}
          />
        </div>
      )}
    </InsightsCard>
  )
}
