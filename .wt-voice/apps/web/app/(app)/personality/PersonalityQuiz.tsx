'use client'

import { useState } from 'react'
import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'

const PRESETS = ['default', 'formal', 'creative', 'analyst', 'empathetic', 'minimal'] as const

const QUESTIONS = [
  {
    key: 'q1',
    options: [
      { label: 'Casually', preset: 'default' },
      { label: 'Formally', preset: 'formal' },
      { label: 'Creatively', preset: 'creative' },
      { label: 'Directly', preset: 'analyst' },
    ],
  },
  {
    key: 'q2',
    options: [
      { label: 'Lots', preset: 'empathetic' },
      { label: 'Some', preset: 'default' },
      { label: 'Minimal', preset: 'analyst' },
      { label: 'Depends', preset: 'creative' },
    ],
  },
  {
    key: 'q3',
    options: [
      { label: 'Very technical', preset: 'analyst' },
      { label: 'Balanced', preset: 'default' },
      { label: 'Simplified', preset: 'empathetic' },
      { label: 'Poetic', preset: 'creative' },
    ],
  },
  {
    key: 'q4',
    options: [
      { label: 'Coding', preset: 'formal' },
      { label: 'Writing', preset: 'creative' },
      { label: 'Analysis', preset: 'analyst' },
      { label: 'Support', preset: 'empathetic' },
    ],
  },
  {
    key: 'q5',
    options: [
      { label: 'Very concise', preset: 'minimal' },
      { label: 'Moderate', preset: 'default' },
      { label: 'Detailed', preset: 'analyst' },
      { label: 'Expansive', preset: 'creative' },
    ],
  },
] as const

interface Props {
  onApply: (preset: string) => void
}

export function PersonalityQuiz({ onApply }: Props) {
  const { t } = useLocale()
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const allAnswered = QUESTIONS.every((q) => answers[q.key])

  const handleAnswer = (questionKey: string, preset: string) => {
    setAnswers((prev) => ({ ...prev, [questionKey]: preset }))
  }

  const computeRecommendation = (): string => {
    const counts: Record<string, number> = {}
    for (const q of QUESTIONS) {
      const preset = answers[q.key]
      if (preset) {
        counts[preset] = (counts[preset] || 0) + 1
      }
    }
    let best = 'default'
    let bestCount = 0
    for (const [preset, count] of Object.entries(counts)) {
      if (count > bestCount) {
        best = preset
        bestCount = count
      }
    }
    return best
  }

  const recommended = allAnswered ? computeRecommendation() : null

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t('personality.quiz.title')}</CardTitle>
        <CardDescription>{t('personality.quiz.subtitle')}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {QUESTIONS.map((q, qi) => (
          <div key={q.key} className="space-y-2">
            <div className="text-sm font-medium">
              {qi + 1}. {t(`personality.quiz.${q.key}`)}
            </div>
            <div className="grid grid-cols-2 gap-2">
              {q.options.map((opt) => (
                <label
                  key={opt.preset}
                  className={`flex items-center gap-2 rounded-md border p-2 text-sm cursor-pointer transition-colors ${
                    answers[q.key] === opt.preset
                      ? 'border-primary bg-primary/5'
                      : 'border-muted hover:border-primary/50'
                  }`}
                >
                  <input
                    type="radio"
                    name={q.key}
                    value={opt.preset}
                    checked={answers[q.key] === opt.preset}
                    onChange={() => handleAnswer(q.key, opt.preset)}
                    className="accent-primary"
                  />
                  {opt.label}
                </label>
              ))}
            </div>
          </div>
        ))}

        {recommended && (
          <div className="flex items-center gap-3 rounded-md border border-primary/30 bg-primary/5 p-4">
            <div className="flex-1">
              <div className="text-sm font-medium">{t('personality.quiz.recommendation')}</div>
              <div className="text-lg font-bold capitalize">{recommended}</div>
            </div>
            <Button onClick={() => onApply(recommended)}>
              {t('personality.quiz.apply')}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
