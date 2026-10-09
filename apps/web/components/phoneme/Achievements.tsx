'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import type { HistoryEntry } from '@/lib/phoneme-store'

interface Achievement {
  id: string
  title: string
  description: string
  icon: string
  category: string
  condition: (state: { history: HistoryEntry[]; quizTotal: number; pronunciationBestStreak: number; flashcardSRS: Record<string, unknown>; challenge?: { score: number } }) => boolean
}

const ACHIEVEMENTS: Achievement[] = [
  {
    id: 'first-word',
    title: 'First Steps',
    description: 'Encode your first word',
    icon: '1',
    category: 'Getting Started',
    condition: (s) => s.history.length >= 1,
  },
  {
    id: 'ten-words',
    title: 'Getting Started',
    description: 'Practice 10 words',
    icon: '10',
    category: 'Getting Started',
    condition: (s) => s.history.length >= 10,
  },
  {
    id: 'fifty-words',
    title: 'Half Century',
    description: 'Practice 50 words',
    icon: '50',
    category: 'Getting Started',
    condition: (s) => s.history.length >= 50,
  },
  {
    id: 'hundred-words',
    title: 'Century',
    description: 'Practice 100 words',
    icon: '100',
    category: 'Getting Started',
    condition: (s) => s.history.length >= 100,
  },
  {
    id: 'perfect-score',
    title: 'Perfection',
    description: 'Get a perfect score (100%)',
    icon: '★',
    category: 'Scores',
    condition: (s) => s.history.some(e => e.score >= 0.99),
  },
  {
    id: 'streak-3',
    title: 'Hat Trick',
    description: 'Get 3 perfect scores in a row',
    icon: '3',
    category: 'Streaks',
    condition: (s) => (s.pronunciationBestStreak ?? 0) >= 3,
  },
  {
    id: 'streak-5',
    title: 'Hot Streak',
    description: 'Get 5 perfect scores in a row',
    icon: '5',
    category: 'Streaks',
    condition: (s) => (s.pronunciationBestStreak ?? 0) >= 5,
  },
  {
    id: 'streak-10',
    title: 'On Fire',
    description: 'Get 10 perfect scores in a row',
    icon: '10',
    category: 'Streaks',
    condition: (s) => (s.pronunciationBestStreak ?? 0) >= 10,
  },
  {
    id: 'multi-lang',
    title: 'Polyglot',
    description: 'Practice words in 3+ languages',
    icon: '🌍',
    category: 'Languages',
    condition: (s) => new Set(s.history.map(e => e.language)).size >= 3,
  },
  {
    id: 'quiz-master',
    title: 'Quiz Master',
    description: 'Complete 20 quizzes',
    icon: 'Q',
    category: 'Quizzes',
    condition: (s) => s.quizTotal >= 20,
  },
  {
    id: 'flashcard-pro',
    title: 'Flashcard Pro',
    description: 'Study 30 flashcards',
    icon: 'F',
    category: 'Flashcards',
    condition: (s) => {
      const srs = s.flashcardSRS ?? {}
      return Object.keys(srs).length >= 30
    },
  },
  {
    id: 'challenge-champion',
    title: 'Challenge Champion',
    description: 'Score 15+ points in daily challenge',
    icon: 'C',
    category: 'Challenges',
    condition: (s) => (s.challenge?.score ?? 0) >= 15,
  },
]

export default function Achievements() {
  const state = usePhonemeStore(s => s)

  const { unlocked, locked } = useMemo(() => {
    const unlocked = ACHIEVEMENTS.filter(a => a.condition(state))
    const locked = ACHIEVEMENTS.filter(a => !a.condition(state))
    return { unlocked, locked }
  }, [state])

  const grouped = useMemo(() => {
    const map: Record<string, { unlocked: Achievement[]; locked: Achievement[] }> = {}
    const all = [...unlocked, ...locked]
    all.forEach(a => {
      if (!map[a.category]) map[a.category] = { unlocked: [], locked: [] }
      map[a.category][a.condition(state) ? 'unlocked' : 'locked'].push(a)
    })
    return map
  }, [unlocked, locked, state])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Achievements</span>
          <div className="flex items-center gap-2">
            <Badge variant="default">{unlocked.length}/{ACHIEVEMENTS.length}</Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-4 sm:grid-cols-6 gap-3 text-center">
          {ACHIEVEMENTS.slice(0, 6).map(a => {
            const isUnlocked = a.condition(state)
            return (
              <div
                key={a.id}
                className={`p-2 rounded-lg border ${
                  isUnlocked
                    ? 'bg-primary/10 border-primary/30'
                    : 'bg-muted/30 border-muted opacity-50'
                }`}
              >
                <div className="text-lg">{a.icon}</div>
                <p className="text-[10px] text-muted-foreground mt-1 leading-tight">{a.title}</p>
              </div>
            )
          })}
        </div>

        {Object.entries(grouped).map(([category, { unlocked: catUnlocked, locked: catLocked }]) => (
          <Collapsible key={category}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
                <span className="font-medium">{category}</span>
                <Badge variant="outline">{catUnlocked.length}/{catUnlocked.length + catLocked.length}</Badge>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-2 space-y-2">
                {[...catUnlocked, ...catLocked].map(a => {
                  const isUnlocked = a.condition(state)
                  return (
                    <div
                      key={a.id}
                      className={`flex items-center gap-3 p-2 rounded ${
                        isUnlocked ? 'bg-primary/5' : 'bg-muted/20 opacity-60'
                      }`}
                    >
                      <span className="text-lg w-8 text-center">{a.icon}</span>
                      <div className="flex-1">
                        <p className="text-sm font-medium">{a.title}</p>
                        <p className="text-xs text-muted-foreground">{a.description}</p>
                      </div>
                      {isUnlocked && <Badge variant="default" className="text-xs">Unlocked</Badge>}
                    </div>
                  )
                })}
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  )
}
