'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { PHONEME_LANGUAGES } from '@/lib/phoneme-controller'

interface Proficiency {
  language: string
  label: string
  level: number
  xp: number
  nextLevelXp: number
  wordsPracticed: number
  avgScore: number
  streak: number
}

function getLevelLabel(level: number): string {
  if (level >= 10) return 'Master'
  if (level >= 7) return 'Advanced'
  if (level >= 4) return 'Intermediate'
  if (level >= 2) return 'Beginner'
  return 'Novice'
}

function getLevelColor(level: number): string {
  if (level >= 7) return 'text-green-600'
  if (level >= 4) return 'text-blue-600'
  if (level >= 2) return 'text-yellow-600'
  return 'text-muted-foreground'
}

export default function ProficiencyTracker() {
  const history = usePhonemeStore(s => s.history)

  const proficiencies = useMemo(() => {
    const langData: Record<string, { scores: number[]; count: number; streak: number }> = {}

    history.forEach(entry => {
      const lang = entry.language
      if (!langData[lang]) langData[lang] = { scores: [], count: 0, streak: 0 }
      langData[lang].scores.push(...entry.scores)
      langData[lang].count++
    })

    return Object.entries(langData)
      .map(([lang, data]) => {
        const avgScore = data.scores.reduce((a, b) => a + b, 0) / data.scores.length
        const xp = Math.round(avgScore * 100 * data.count)
        const level = Math.min(10, Math.floor(1 + Math.log2(data.count + 1) + avgScore * 3))
        const nextLevelXp = Math.round(Math.pow(2, level) * 10)

        return {
          language: lang,
          label: PHONEME_LANGUAGES.find(l => l.value === lang)?.label || lang,
          level,
          xp,
          nextLevelXp,
          wordsPracticed: data.count,
          avgScore,
          streak: data.streak,
        }
      })
      .sort((a, b) => b.level - a.level || b.xp - a.xp)
  }, [history])

  if (proficiencies.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Language Proficiency</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Start practicing to see your proficiency levels.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Language Proficiency</span>
          <Badge variant="outline">{proficiencies.length} languages</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {proficiencies.map(p => (
          <div key={p.language} className="p-3 rounded-lg bg-muted/20 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-medium">{p.label}</span>
                <Badge variant="outline" className={getLevelColor(p.level)}>
                  Lv.{p.level}
                </Badge>
              </div>
              <span className="text-xs text-muted-foreground">{getLevelLabel(p.level)}</span>
            </div>

            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span>{p.wordsPracticed} words</span>
              <span>{(p.avgScore * 100).toFixed(0)}% avg</span>
              <span>{p.xp} XP</span>
            </div>

            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-primary"
                style={{ width: `${Math.min(100, (p.xp / p.nextLevelXp) * 100)}%` }}
              />
            </div>
            <p className="text-[10px] text-muted-foreground text-right">
              {p.xp}/{p.nextLevelXp} XP to next level
            </p>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
