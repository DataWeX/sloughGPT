'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'

interface SpeedStat {
  language: string
  label: string
  avgTime: number
  fastestTime: number
  slowestTime: number
  attempts: number
  wordsPerMinute: number
}

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export default function PronunciationSpeed() {
  const history = usePhonemeStore(s => s.history)

  const stats = useMemo(() => {
    const langData: Record<string, { times: number[]; count: number }> = {}

    const sorted = [...history].sort((a, b) => a.timestamp - b.timestamp)

    sorted.forEach((entry, i) => {
      const lang = entry.language
      if (!langData[lang]) langData[lang] = { times: [], count: 0 }
      if (i > 0 && sorted[i - 1].language === lang) {
        const gap = entry.timestamp - sorted[i - 1].timestamp
        if (gap > 0 && gap < 120000) {
          langData[lang].times.push(gap)
        }
      }
      langData[lang].count++
    })

    return Object.entries(langData)
      .filter(([, data]) => data.times.length > 0)
      .map(([lang, data]) => {
        const avgTime = data.times.reduce((a, b) => a + b, 0) / data.times.length
        const fastestTime = Math.min(...data.times)
        const slowestTime = Math.max(...data.times)
        const wordsPerMinute = avgTime > 0 ? 60000 / avgTime : 0

        return {
          language: lang,
          label: lang.toUpperCase(),
          avgTime,
          fastestTime,
          slowestTime,
          attempts: data.count,
          wordsPerMinute,
        }
      })
      .sort((a, b) => a.avgTime - b.avgTime)
  }, [history])

  if (stats.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Practice Speed</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see your speed statistics.
          </p>
        </CardContent>
      </Card>
    )
  }

  const overallAvg = stats.reduce((sum, s) => sum + s.avgTime * s.attempts, 0) / stats.reduce((sum, s) => sum + s.attempts, 0)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Practice Speed</span>
          <Badge variant="outline">Avg: {formatTime(overallAvg)}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
          <div className="p-2 rounded-lg bg-muted/30">
            <p className="text-lg font-bold">{formatTime(overallAvg)}</p>
            <p className="text-xs text-muted-foreground">Overall Avg</p>
          </div>
          <div className="p-2 rounded-lg bg-muted/30">
            <p className="text-lg font-bold">{stats.length}</p>
            <p className="text-xs text-muted-foreground">Languages</p>
          </div>
          <div className="p-2 rounded-lg bg-muted/30">
            <p className="text-lg font-bold">{stats.reduce((sum, s) => sum + s.attempts, 0)}</p>
            <p className="text-xs text-muted-foreground">Total Attempts</p>
          </div>
          <div className="p-2 rounded-lg bg-muted/30">
            <p className="text-lg font-bold">{Math.round(stats.reduce((sum, s) => sum + s.wordsPerMinute * s.attempts, 0) / stats.reduce((sum, s) => sum + s.attempts, 0))}</p>
            <p className="text-xs text-muted-foreground">Words/Min</p>
          </div>
        </div>

        {stats.map(s => (
          <Collapsible key={s.language}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">{s.label}</Badge>
                  <span className="text-xs text-muted-foreground">{s.attempts} attempts</span>
                </div>
                <span className="font-medium">{formatTime(s.avgTime)}</span>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-3 pt-0 space-y-2">
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>Fastest: {formatTime(s.fastestTime)}</span>
                  <span>Slowest: {formatTime(s.slowestTime)}</span>
                  <span>{Math.round(s.wordsPerMinute)} WPM</span>
                </div>
                <div className="h-2 rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-primary"
                    style={{ width: `${Math.min(100, (s.wordsPerMinute / 120) * 100)}%` }}
                  />
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  )
}
