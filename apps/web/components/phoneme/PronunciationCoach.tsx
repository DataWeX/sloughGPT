'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'

interface Tip {
  category: string
  title: string
  description: string
  severity: 'high' | 'medium' | 'low'
}

function analyzePatterns(history: NonNullable<ReturnType<typeof usePhonemeStore.getState>['history']>): Tip[] {
  const tips: Tip[] = []
  if (history.length < 3) return tips

  const phonemeScores: Record<string, number[]> = {}
  history.forEach(entry => {
    entry.targetPhonemes.forEach((p, i) => {
      const score = entry.scores[i] ?? entry.scores[0] ?? 0
      phonemeScores[p] = phonemeScores[p] || []
      phonemeScores[p].push(score)
    })
  })

  const phonemeAvgs = Object.entries(phonemeScores).map(([phoneme, scores]) => ({
    phoneme,
    avg: scores.reduce((a, b) => a + b, 0) / scores.length,
    count: scores.length,
  }))

  const weakPhonemes = phonemeAvgs
    .filter(p => p.count >= 2 && p.avg < 0.7)
    .sort((a, b) => a.avg - b.avg)

  if (weakPhonemes.length > 0) {
    const worst = weakPhonemes[0]
    tips.push({
      category: 'Phoneme Difficulty',
      title: `Struggling with "${worst.phoneme}"`,
      description: `Your average score for "${worst.phoneme}" is ${(worst.avg * 100).toFixed(0)}% across ${worst.count} attempts. Focus on this phoneme in practice.`,
      severity: worst.avg < 0.5 ? 'high' : 'medium',
    })
  }

  const strongPhonemes = phonemeAvgs
    .filter(p => p.count >= 2 && p.avg >= 0.9)
    .sort((a, b) => b.avg - a.avg)

  if (strongPhonemes.length > 0) {
    const best = strongPhonemes[0]
    tips.push({
      category: 'Strength',
      title: `Excellent at "${best.phoneme}"`,
      description: `Your average score for "${best.phoneme}" is ${(best.avg * 100).toFixed(0)}% across ${best.count} attempts. You've mastered this phoneme!`,
      severity: 'low',
    })
  }

  const langScores: Record<string, number[]> = {}
  history.forEach(entry => {
    if (!langScores[entry.language]) langScores[entry.language] = []
    langScores[entry.language].push(entry.scores.reduce((a, b) => a + b, 0) / entry.scores.length)
  })

  const langAvgs = Object.entries(langScores).map(([lang, scores]) => ({
    lang,
    avg: scores.reduce((a, b) => a + b, 0) / scores.length,
    count: scores.length,
  }))

  const weakLangs = langAvgs.filter(l => l.avg < 0.7 && l.count >= 2)
  if (weakLangs.length > 0) {
    tips.push({
      category: 'Language Focus',
      title: `Need more practice in ${weakLangs[0].lang}`,
      description: `Your average score in ${weakLangs[0].lang} is ${(weakLangs[0].avg * 100).toFixed(0)}%. Try the flashcards or practice mode for this language.`,
      severity: 'medium',
    })
  }

  const recent = history.slice(-5)
  const recentAvg = recent.reduce((sum, e) => sum + (e.scores.reduce((a, b) => a + b, 0) / e.scores.length), 0) / recent.length

  if (recentAvg < 0.6) {
    tips.push({
      category: 'Performance',
      title: 'Recent scores are declining',
      description: `Your last 5 attempts average ${(recentAvg * 100).toFixed(0)}%. Try reviewing your flashcards or switching to easier words.`,
      severity: 'high',
    })
  } else if (recentAvg > 0.85) {
    tips.push({
      category: 'Performance',
      title: 'Great momentum!',
      description: `Your last 5 attempts average ${(recentAvg * 100).toFixed(0)}%. Keep it up! Consider trying harder difficulty levels.`,
      severity: 'low',
    })
  }

  if (tips.length === 0) {
    tips.push({
      category: 'General',
      title: 'Keep practicing!',
      description: 'Practice more to get personalized pronunciation tips based on your patterns.',
      severity: 'low',
    })
  }

  return tips
}

function getSeverityBadge(severity: 'high' | 'medium' | 'low') {
  switch (severity) {
    case 'high': return <Badge variant="destructive">Focus</Badge>
    case 'medium': return <Badge variant="secondary">Improve</Badge>
    case 'low': return <Badge variant="outline">Good</Badge>
  }
}

export default function PronunciationCoach() {
  const history = usePhonemeStore(s => s.history)

  const tips = useMemo(() => analyzePatterns(history), [history])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Coach</span>
          <Badge variant="outline">{tips.length} tips</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {tips.map((tip, i) => (
          <Collapsible key={i}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/30 transition-colors text-sm">
                <div className="flex items-center gap-2 text-left">
                  <span className="font-medium">{tip.title}</span>
                  {getSeverityBadge(tip.severity)}
                </div>
                <Badge variant="outline" className="text-xs">{tip.category}</Badge>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-3 pt-0 text-sm text-muted-foreground">
                {tip.description}
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  )
}
