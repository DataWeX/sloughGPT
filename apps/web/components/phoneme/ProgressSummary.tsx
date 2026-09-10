'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { PHONEME_LANGUAGES } from '@/lib/phoneme-controller'

export default function ProgressSummary() {
  const history = usePhonemeStore(s => s.history)
  const pronunciationBestStreak = usePhonemeStore(s => s.pronunciationBestStreak)

  const stats = useMemo(() => {
    if (history.length === 0) {
      return {
        totalAttempts: 0,
        averageScore: 0,
        excellentCount: 0,
        needsWorkCount: 0,
        languageBreakdown: [],
        recentTrend: [],
        topWords: [],
        worstWords: [],
      }
    }

    const totalAttempts = history.length
    const averageScore = history.reduce((sum, h) => sum + h.score, 0) / totalAttempts
    const excellentCount = history.filter(h => h.score >= 0.8).length
    const needsWorkCount = history.filter(h => h.score < 0.5).length

    const languageCounts: Record<string, number> = {}
    history.forEach(h => {
      languageCounts[h.language] = (languageCounts[h.language] || 0) + 1
    })
    const languageBreakdown = Object.entries(languageCounts)
      .map(([code, count]) => ({
        code,
        label: PHONEME_LANGUAGES.find(l => l.value === code)?.label ?? code.toUpperCase(),
        count,
        percentage: (count / totalAttempts) * 100,
      }))
      .sort((a, b) => b.count - a.count)

    const last20 = history.slice(-20)
    const recentTrend = last20.map((h, i) => ({
      index: i + 1,
      score: Math.round(h.score * 100),
      word: h.target,
    }))

    const wordScores: Record<string, { total: number; count: number }> = {}
    history.forEach(h => {
      if (!wordScores[h.target]) wordScores[h.target] = { total: 0, count: 0 }
      wordScores[h.target].total += h.score
      wordScores[h.target].count++
    })

    const wordAvg = Object.entries(wordScores).map(([word, data]) => ({
      word,
      avgScore: data.total / data.count,
      attempts: data.count,
    }))

    const topWords = wordAvg
      .filter(w => w.attempts >= 2)
      .sort((a, b) => b.avgScore - a.avgScore)
      .slice(0, 5)

    const worstWords = wordAvg
      .filter(w => w.attempts >= 2)
      .sort((a, b) => a.avgScore - b.avgScore)
      .slice(0, 5)

    return {
      totalAttempts,
      averageScore,
      excellentCount,
      needsWorkCount,
      languageBreakdown,
      recentTrend,
      topWords,
      worstWords,
    }
  }, [history])

  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Progress Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-8">
            No practice data yet. Start practicing to see your progress!
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Progress Summary</span>
          <Badge variant="outline">{stats.totalAttempts} attempts</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold">{(stats.averageScore * 100).toFixed(0)}%</p>
            <p className="text-xs text-muted-foreground">Average Score</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-success">{stats.excellentCount}</p>
            <p className="text-xs text-muted-foreground">Excellent (≥80%)</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-destructive">{stats.needsWorkCount}</p>
            <p className="text-xs text-muted-foreground">Needs Work (&lt;50%)</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{pronunciationBestStreak}</p>
            <p className="text-xs text-muted-foreground">Best Streak</p>
          </div>
        </div>

        {stats.recentTrend.length > 1 && (
          <div>
            <p className="text-sm font-medium text-muted-foreground mb-2">Recent Trend</p>
            <div className="h-[120px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stats.recentTrend}>
                  <defs>
                    <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="index" hide />
                  <YAxis domain={[0, 100]} hide />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const data = payload[0].payload
                      return (
                        <div className="rounded-lg border bg-background p-2 shadow-sm">
                          <p className="text-sm font-medium">{data.word}</p>
                          <p className="text-xs text-muted-foreground">{data.score}%</p>
                        </div>
                      )
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="score"
                    stroke="hsl(var(--primary))"
                    fill="url(#scoreGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {stats.languageBreakdown.length > 0 && (
          <div>
            <p className="text-sm font-medium text-muted-foreground mb-2">Language Breakdown</p>
            <div className="space-y-2">
              {stats.languageBreakdown.map(lang => (
                <div key={lang.code} className="flex items-center gap-3">
                  <span className="text-sm w-20">{lang.label}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${lang.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-12 text-right">
                    {lang.count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {stats.topWords.length > 0 && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium text-muted-foreground mb-2">Strongest Words</p>
              <div className="space-y-1">
                {stats.topWords.map(w => (
                  <div key={w.word} className="flex items-center justify-between text-sm">
                    <span>{w.word}</span>
                    <Badge variant="default" className="bg-success">{(w.avgScore * 100).toFixed(0)}%</Badge>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground mb-2">Words to Practice</p>
              <div className="space-y-1">
                {stats.worstWords.map(w => (
                  <div key={w.word} className="flex items-center justify-between text-sm">
                    <span>{w.word}</span>
                    <Badge variant="destructive">{(w.avgScore * 100).toFixed(0)}%</Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
