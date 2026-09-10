'use client'

import { useMemo, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { PHONEME_LANGUAGES } from '@/lib/phoneme-controller'
import type { PhonemeLanguage } from '@/lib/phoneme-controller'

interface WordStats {
  word: string
  language: string
  attempts: number
  avgScore: number
  bestScore: number
  worstScore: number
  lastAttempt: number
  difficulty: 'easy' | 'medium' | 'hard'
}

function getDifficulty(avg: number): 'easy' | 'medium' | 'hard' {
  if (avg >= 0.8) return 'easy'
  if (avg >= 0.5) return 'medium'
  return 'hard'
}

function getDifficultyColor(d: string): string {
  switch (d) {
    case 'easy': return 'bg-green-100 text-green-800'
    case 'medium': return 'bg-yellow-100 text-yellow-800'
    case 'hard': return 'bg-red-100 text-red-800'
    default: return ''
  }
}

export default function WordDifficultyRanker() {
  const history = usePhonemeStore(s => s.history)
  const [filter, setFilter] = useState<'all' | 'easy' | 'medium' | 'hard'>('all')
  const [sortBy, setSortBy] = useState<'difficulty' | 'attempts' | 'score'>('difficulty')

  const wordStats = useMemo(() => {
    const wordMap: Record<string, WordStats> = {}

    history.forEach(entry => {
      const key = `${entry.target}-${entry.language}`
      if (!wordMap[key]) {
        wordMap[key] = {
          word: entry.target,
          language: entry.language,
          attempts: 0,
          avgScore: 0,
          bestScore: 0,
          worstScore: 1,
          lastAttempt: entry.timestamp,
          difficulty: 'medium',
        }
      }
      const w = wordMap[key]
      w.attempts++
      w.avgScore = (w.avgScore * (w.attempts - 1) + entry.score) / w.attempts
      w.bestScore = Math.max(w.bestScore, entry.score)
      w.worstScore = Math.min(w.worstScore, entry.score)
      if (entry.timestamp > w.lastAttempt) w.lastAttempt = entry.timestamp
    })

    return Object.values(wordMap).map(w => ({
      ...w,
      difficulty: getDifficulty(w.avgScore),
    }))
  }, [history])

  const ranked = useMemo(() => {
    let result = [...wordStats]
    if (filter !== 'all') result = result.filter(w => w.difficulty === filter)
    result.sort((a, b) => {
      if (sortBy === 'difficulty') {
        const order = { hard: 0, medium: 1, easy: 2 }
        return order[a.difficulty] - order[b.difficulty] || a.avgScore - b.avgScore
      }
      if (sortBy === 'attempts') return b.attempts - a.attempts
      return a.avgScore - b.avgScore
    })
    return result
  }, [wordStats, filter, sortBy])

  if (wordStats.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Word Difficulty</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more words to see difficulty rankings.
          </p>
        </CardContent>
      </Card>
    )
  }

  const hardCount = wordStats.filter(w => w.difficulty === 'hard').length
  const medCount = wordStats.filter(w => w.difficulty === 'medium').length
  const easyCount = wordStats.filter(w => w.difficulty === 'easy').length

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Word Difficulty</span>
          <Badge variant="outline">{wordStats.length} words</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2 flex-wrap">
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">Filter:</span>
            {(['all', 'hard', 'medium', 'easy'] as const).map(d => (
              <Badge
                key={d}
                variant={filter === d ? 'default' : 'outline'}
                className="cursor-pointer text-xs"
                onClick={() => setFilter(d)}
              >
                {d === 'all' ? `All (${wordStats.length})` : `${d} (${d === 'hard' ? hardCount : d === 'medium' ? medCount : easyCount})`}
              </Badge>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">Sort:</span>
            {(['difficulty', 'attempts', 'score'] as const).map(s => (
              <Badge
                key={s}
                variant={sortBy === s ? 'default' : 'outline'}
                className="cursor-pointer text-xs"
                onClick={() => setSortBy(s)}
              >
                {s}
              </Badge>
            ))}
          </div>
        </div>

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Words ({ranked.length})</span>
              <Badge variant="outline">{sortBy}</Badge>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-2">
              {ranked.map((w, i) => (
                <div key={`${w.word}-${w.language}`} className="flex items-center gap-3 p-2 rounded bg-muted/20">
                  <span className="text-xs text-muted-foreground w-4">{i + 1}</span>
                  <span className="font-medium text-sm min-w-[80px]">{w.word}</span>
                  <Badge variant="outline" className="text-xs">{w.language}</Badge>
                  <Badge className={`text-xs ${getDifficultyColor(w.difficulty)}`}>
                    {w.difficulty}
                  </Badge>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        w.difficulty === 'easy' ? 'bg-green-500' :
                        w.difficulty === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${w.avgScore * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-20 text-right">
                    {(w.avgScore * 100).toFixed(0)}% ({w.attempts}x)
                  </span>
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  )
}
