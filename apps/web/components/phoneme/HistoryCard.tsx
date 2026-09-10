'use client'

import { useState, useCallback, useRef, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconTrash, IconDownload, IconUpload } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Progress } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore, type HistoryEntry } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'
import { PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

function HistoryRow({ entry }: { entry: HistoryEntry }) {
  const time = new Date(entry.timestamp).toLocaleTimeString()
  const pct = (entry.score * 100).toFixed(0)

  return (
    <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 text-sm">
      <span className="text-muted-foreground w-16 shrink-0">{time}</span>
      <span className="font-medium">{entry.target}</span>
      <span className="text-muted-foreground">→</span>
      <span>{entry.spoken}</span>
      <Badge
        variant={entry.score >= 0.8 ? 'default' : entry.score >= 0.5 ? 'secondary' : 'destructive'}
        className="ml-auto"
      >
        {pct}%
      </Badge>
      <Badge variant="outline" className="w-8 justify-center">{entry.language.toUpperCase()}</Badge>
    </div>
  )
}

function ProgressChart({ history }: { history: HistoryEntry[] }) {
  const chartData = useMemo(() => {
    if (history.length < 2) return []
    const sorted = [...history].sort((a, b) => a.timestamp - b.timestamp)
    return sorted.map((entry, i) => ({
      index: i + 1,
      score: Math.round(entry.score * 100),
      word: entry.target,
      time: new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }))
  }, [history])

  if (chartData.length < 2) return null

  return (
    <Collapsible>
      <CollapsibleTrigger className="w-full">
        <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
          <span className="font-medium">Progress</span>
          <span className="text-muted-foreground">{chartData.length} data points</span>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="p-2 h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgb(var(--primary))" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="rgb(var(--primary))" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" />
              <XAxis
                dataKey="index"
                tick={{ fontSize: 10, fill: 'rgb(var(--muted-foreground))' }}
                axisLine={{ stroke: 'rgb(var(--border))' }}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 10, fill: 'rgb(var(--muted-foreground))' }}
                axisLine={{ stroke: 'rgb(var(--border))' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgb(var(--card))',
                  border: '1px solid rgb(var(--border))',
                  borderRadius: '0.5rem',
                  fontSize: '0.75rem',
                }}
                formatter={(value: number) => [`${value}%`, 'Score']}
                labelFormatter={(label) => `#${label}`}
              />
              <Area
                type="monotone"
                dataKey="score"
                stroke="rgb(var(--primary))"
                strokeWidth={2}
                fill="url(#scoreGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function HistoryStats({ history }: { history: HistoryEntry[] }) {
  const stats = useMemo(() => {
    if (history.length === 0) return null
    const avgScore = history.reduce((sum, e) => sum + e.score, 0) / history.length
    const langCounts: Record<string, number> = {}
    let excellent = 0, good = 0, needsWork = 0
    for (const e of history) {
      langCounts[e.language] = (langCounts[e.language] || 0) + 1
      if (e.score >= 0.8) excellent++
      else if (e.score >= 0.5) good++
      else needsWork++
    }
    const topLang = Object.entries(langCounts).sort((a, b) => b[1] - a[1])[0]
    return { avgScore, langCounts, excellent, good, needsWork, topLang, total: history.length }
  }, [history])

  if (!stats) return null

  return (
    <Collapsible>
      <CollapsibleTrigger className="w-full">
        <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
          <span className="font-medium">Stats</span>
          <span className="text-muted-foreground">{stats.total} entries · avg {(stats.avgScore * 100).toFixed(0)}%</span>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-2">
          <div className="text-center p-2 rounded-lg bg-background">
            <p className="text-lg font-bold">{stats.total}</p>
            <p className="text-xs text-muted-foreground">Total</p>
          </div>
          <div className="text-center p-2 rounded-lg bg-background">
            <p className="text-lg font-bold">{(stats.avgScore * 100).toFixed(0)}%</p>
            <p className="text-xs text-muted-foreground">Avg Score</p>
          </div>
          <div className="text-center p-2 rounded-lg bg-background">
            <p className="text-lg font-bold text-success">{stats.excellent}</p>
            <p className="text-xs text-muted-foreground">≥80%</p>
          </div>
          <div className="text-center p-2 rounded-lg bg-background">
            <p className="text-lg font-bold text-destructive">{stats.needsWork}</p>
            <p className="text-xs text-muted-foreground">&lt;50%</p>
          </div>
        </div>
        <div className="px-2 pb-2">
          <p className="text-xs text-muted-foreground mb-1">Score Distribution</p>
          <div className="flex gap-1 h-2">
            <div className="bg-success rounded-l" style={{ flex: stats.excellent }} />
            <div className="bg-primary" style={{ flex: stats.good }} />
            <div className="bg-destructive rounded-r" style={{ flex: stats.needsWork }} />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>≥80% ({stats.excellent})</span>
            <span>50-79% ({stats.good})</span>
            <span>&lt;50% ({stats.needsWork})</span>
          </div>
        </div>
        {stats.topLang && (
          <div className="px-2 pb-2">
            <p className="text-xs text-muted-foreground">Most practiced: <Badge variant="outline" className="ml-1">{stats.topLang[0].toUpperCase()}</Badge> ({stats.topLang[1]} entries)</p>
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}

function WordDifficulty({ history }: { history: HistoryEntry[] }) {
  const wordStats = useMemo(() => {
    if (history.length === 0) return null
    const words: Record<string, { total: number; sum: number; lang: string }> = {}
    for (const e of history) {
      const key = `${e.language}:${e.target.toLowerCase()}`
      if (!words[key]) words[key] = { total: 0, sum: 0, lang: e.language }
      words[key].total++
      words[key].sum += e.score
    }
    const ranked = Object.entries(words)
      .map(([key, s]) => ({
        word: key.split(':')[1],
        lang: s.lang,
        avg: s.sum / s.total,
        attempts: s.total,
      }))
      .sort((a, b) => a.avg - b.avg)
    return { hardest: ranked.slice(0, 5), easiest: ranked.slice(-5).reverse(), total: ranked.length }
  }, [history])

  if (!wordStats || wordStats.total === 0) return null

  return (
    <Collapsible>
      <CollapsibleTrigger className="w-full">
        <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
          <span className="font-medium">Word Difficulty</span>
          <span className="text-muted-foreground">{wordStats.total} unique words</span>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-2">
          <div>
            <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
              <span className="inline-block w-2 h-2 rounded-full bg-destructive" />
              Hardest
            </p>
            {wordStats.hardest.map((w, i) => (
              <div key={i} className="flex items-center justify-between py-1 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{w.word}</span>
                  <Badge variant="outline" className="text-xs">{w.lang.toUpperCase()}</Badge>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="destructive" className="text-xs">{(w.avg * 100).toFixed(0)}%</Badge>
                  <span className="text-xs text-muted-foreground">×{w.attempts}</span>
                </div>
              </div>
            ))}
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
              <span className="inline-block w-2 h-2 rounded-full bg-success" />
              Easiest
            </p>
            {wordStats.easiest.map((w, i) => (
              <div key={i} className="flex items-center justify-between py-1 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{w.word}</span>
                  <Badge variant="outline" className="text-xs">{w.lang.toUpperCase()}</Badge>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="default" className="text-xs">{(w.avg * 100).toFixed(0)}%</Badge>
                  <span className="text-xs text-muted-foreground">×{w.attempts}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

export default function HistoryCard() {
  const [filterLang, setFilterLang] = useState<string>('all')
  const history = usePhonemeStore(s => s.history)
  const clearHistory = usePhonemeStore(s => s.clearHistory)
  const exportHistory = usePhonemeStore(s => s.exportHistory)
  const importHistory = usePhonemeStore(s => s.importHistory)
  const addToast = useToastStore(s => s.addToast)
  const fileRef = useRef<HTMLInputElement>(null)

  const filteredHistory = useMemo(() => {
    if (filterLang === 'all') return history
    return history.filter(e => e.language === filterLang)
  }, [history, filterLang])

  const handleExport = useCallback(() => {
    const json = exportHistory()
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'phoneme-history.json'
    a.click()
    URL.revokeObjectURL(url)
    addToast('History exported as JSON', 'success')
  }, [exportHistory, addToast])

  const handleExportCSV = useCallback(() => {
    const headers = ['Time', 'Target', 'Spoken', 'Score', 'Language', 'Target Phonemes', 'Spoken Phonemes']
    const rows = history.map(e => [
      new Date(e.timestamp).toISOString(),
      e.target,
      e.spoken,
      (e.score * 100).toFixed(1),
      e.language,
      e.targetPhonemes.join(' '),
      e.spokenPhonemes.join(' '),
    ])
    const csv = [headers, ...rows].map(r => r.map(c => `"${c}"`).join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'phoneme-history.csv'
    a.click()
    URL.revokeObjectURL(url)
    addToast('History exported as CSV', 'success')
  }, [history, addToast])

  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const json = reader.result as string
      importHistory(json)
      addToast('History imported', 'success')
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [importHistory, addToast])

  const handleClear = useCallback(() => {
    clearHistory()
    addToast('History cleared', 'info')
  }, [clearHistory, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation History</span>
          <span className="text-sm font-normal text-muted-foreground">{filteredHistory.length} entries</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <HistoryStats history={history} />
        <ProgressChart history={history} />
        <WordDifficulty history={history} />

        <div className="flex flex-col sm:flex-row gap-2">
          <Select value={filterLang} onValueChange={setFilterLang}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Languages</SelectItem>
              {PHONEME_LANGUAGES.map(l => (
                <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="flex gap-2">
            <Button variant="secondary" size="sm" onClick={handleExport}>
              <IconDownload className="mr-2 h-4 w-4" />
              JSON
            </Button>
            <Button variant="secondary" size="sm" onClick={handleExportCSV}>
              <IconDownload className="mr-2 h-4 w-4" />
              CSV
            </Button>
            <Button variant="secondary" size="sm" onClick={() => fileRef.current?.click()}>
              <IconUpload className="mr-2 h-4 w-4" />
              Import
            </Button>
            <Button variant="destructive" size="sm" onClick={handleClear}>
              <IconTrash className="mr-2 h-4 w-4" />
              Clear
            </Button>
          </div>
          <input ref={fileRef} type="file" accept=".json" className="hidden" onChange={handleImport} />
        </div>

        <div className="space-y-1 max-h-[400px] overflow-y-auto">
          {filteredHistory.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              {history.length === 0
                ? 'No history yet. Score some pronunciation to get started.'
                : 'No entries match the selected filter.'}
            </p>
          ) : (
            filteredHistory.map((entry, i) => (
              <div key={entry.id} className="animate-in fade-in duration-200" style={{ animationDelay: `${Math.min(i * 30, 300)}ms` }}>
                <HistoryRow entry={entry} />
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}
