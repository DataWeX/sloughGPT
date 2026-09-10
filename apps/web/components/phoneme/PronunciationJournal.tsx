'use client'

import { useState, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconTrash, IconDownload } from '@sloughgpt/strui'
import { usePhonemeStore, type HistoryEntry } from '@/lib/phoneme-store'
import { PHONEME_LANGUAGES } from '@/lib/phoneme-controller'

interface JournalEntry {
  date: string
  entries: HistoryEntry[]
  totalAttempts: number
  averageScore: number
  bestStreak: number
}

const JOURNAL_KEY = 'sloughgpt-phoneme-journal'

function loadJournal(): JournalEntry[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(JOURNAL_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveJournal(journal: JournalEntry[]) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(JOURNAL_KEY, JSON.stringify(journal))
  } catch {}
}

export default function PronunciationJournal() {
  const history = usePhonemeStore(s => s.history)
  const [journal, setJournal] = useState<JournalEntry[]>(() => loadJournal())

  const dailyEntries = useMemo(() => {
    const grouped: Record<string, HistoryEntry[]> = {}
    history.forEach(entry => {
      const date = new Date(entry.timestamp).toISOString().split('T')[0]
      if (!grouped[date]) grouped[date] = []
      grouped[date].push(entry)
    })

    return Object.entries(grouped)
      .map(([date, entries]) => ({
        date,
        entries,
        totalAttempts: entries.length,
        averageScore: entries.reduce((sum, e) => sum + e.score, 0) / entries.length,
        bestStreak: entries.reduce((best, e, i) => {
          if (e.score >= 0.8) {
            const streak = entries.slice(0, i + 1).filter(x => x.score >= 0.8).length
            return Math.max(best, streak)
          }
          return best
        }, 0),
      }))
      .sort((a, b) => b.date.localeCompare(a.date))
  }, [history])

  const handleSaveToJournal = useCallback(() => {
    const today = new Date().toISOString().split('T')[0]
    const todayEntries = history.filter(e => 
      new Date(e.timestamp).toISOString().split('T')[0] === today
    )
    
    if (todayEntries.length === 0) return

    const entry: JournalEntry = {
      date: today,
      entries: todayEntries,
      totalAttempts: todayEntries.length,
      averageScore: todayEntries.reduce((sum, e) => sum + e.score, 0) / todayEntries.length,
      bestStreak: todayEntries.reduce((best, e, i) => {
        if (e.score >= 0.8) {
          const streak = todayEntries.slice(0, i + 1).filter(x => x.score >= 0.8).length
          return Math.max(best, streak)
        }
        return best
      }, 0),
    }

    const existingIndex = journal.findIndex(j => j.date === today)
    let newJournal
    if (existingIndex >= 0) {
      newJournal = [...journal]
      newJournal[existingIndex] = entry
    } else {
      newJournal = [entry, ...journal]
    }
    
    setJournal(newJournal)
    saveJournal(newJournal)
  }, [history, journal])

  const handleDeleteEntry = useCallback((date: string) => {
    const newJournal = journal.filter(j => j.date !== date)
    setJournal(newJournal)
    saveJournal(newJournal)
  }, [journal])

  const handleExport = useCallback(() => {
    const data = JSON.stringify(journal, null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `pronunciation-journal-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [journal])

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-success'
    if (score >= 0.5) return 'text-warning'
    return 'text-destructive'
  }

  const getScoreBadge = (score: number) => {
    if (score >= 0.8) return 'default'
    if (score >= 0.5) return 'secondary'
    return 'destructive'
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Journal</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleSaveToJournal}>
              Save Today
            </Button>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <IconDownload className="h-4 w-4" />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {dailyEntries.length === 0 ? (
          <p className="text-muted-foreground text-center py-4">
            No practice data yet. Start practicing to build your journal!
          </p>
        ) : (
          <div className="space-y-3">
            {dailyEntries.map(day => (
              <div key={day.date} className="p-3 rounded-lg bg-muted/30 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{day.date}</span>
                    <Badge variant={getScoreBadge(day.averageScore)}>
                      {(day.averageScore * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {day.totalAttempts} attempts
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteEntry(day.date)}
                    >
                      <IconTrash className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-1">
                  {day.entries.slice(0, 10).map((entry, i) => (
                    <Badge key={i} variant="outline" className="text-xs">
                      {entry.target}: {(entry.score * 100).toFixed(0)}%
                    </Badge>
                  ))}
                  {day.entries.length > 10 && (
                    <Badge variant="outline" className="text-xs">
                      +{day.entries.length - 10} more
                    </Badge>
                  )}
                </div>

                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>Words practiced: {new Set(day.entries.map(e => e.target)).size}</span>
                  <span>Languages: {[...new Set(day.entries.map(e => e.language))].join(', ')}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="p-3 rounded-lg bg-muted/30">
          <p className="text-sm text-muted-foreground">
            <span className="font-medium">Journal stats:</span> {journal.length} days logged
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
