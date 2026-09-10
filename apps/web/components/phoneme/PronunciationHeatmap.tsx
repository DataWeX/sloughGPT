'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { toIPA } from '@/lib/phoneme-controller'

interface PhonemeAccuracy {
  phoneme: string
  ipa: string
  total: number
  correct: number
  accuracy: number
  category: string
}

function getPhonemeCategory(phoneme: string): string {
  const vowels = ['IY', 'IH', 'EY', 'EH', 'AE', 'AA', 'AH', 'AO', 'OW', 'OY', 'UH', 'UW', 'ER', 'AX']
  const stops = ['P', 'B', 'T', 'D', 'K', 'G']
  const fricatives = ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH']
  const nasals = ['M', 'N', 'NG']
  const liquids = ['L', 'R']
  const glides = ['W', 'Y']
  const affricates = ['CH', 'JH']

  if (vowels.includes(phoneme)) return 'Vowels'
  if (stops.includes(phoneme)) return 'Stops'
  if (fricatives.includes(phoneme)) return 'Fricatives'
  if (nasals.includes(phoneme)) return 'Nasals'
  if (liquids.includes(phoneme)) return 'Liquids'
  if (glides.includes(phoneme)) return 'Glides'
  if (affricates.includes(phoneme)) return 'Affricates'
  return 'Other'
}

function getAccuracyColor(accuracy: number): string {
  if (accuracy >= 0.9) return 'bg-green-500'
  if (accuracy >= 0.7) return 'bg-yellow-500'
  if (accuracy >= 0.5) return 'bg-orange-500'
  if (accuracy >= 0.3) return 'bg-red-400'
  return 'bg-red-600'
}

function getAccuracyTextColor(accuracy: number): string {
  if (accuracy >= 0.9) return 'text-green-700'
  if (accuracy >= 0.7) return 'text-yellow-700'
  if (accuracy >= 0.5) return 'text-orange-700'
  if (accuracy >= 0.3) return 'text-red-500'
  return 'text-red-700'
}

const CATEGORIES = ['Vowels', 'Stops', 'Fricatives', 'Affricates', 'Nasals', 'Liquids', 'Glides']

export default function PronunciationHeatmap() {
  const history = usePhonemeStore(s => s.history)

  const phonemeData = useMemo(() => {
    if (history.length === 0) return []

    const phonemeStats: Record<string, { total: number; correct: number }> = {}

    history.forEach(entry => {
      entry.targetPhonemes.forEach((phoneme, i) => {
        if (!phonemeStats[phoneme]) phonemeStats[phoneme] = { total: 0, correct: 0 }
        phonemeStats[phoneme].total++
        const score = entry.scores[i] ?? entry.scores[0] ?? 0
        if (score >= 0.8) phonemeStats[phoneme].correct++
      })
    })

    return Object.entries(phonemeStats)
      .map(([phoneme, stats]) => ({
        phoneme,
        ipa: toIPA([phoneme])[0] || phoneme,
        total: stats.total,
        correct: stats.correct,
        accuracy: stats.correct / stats.total,
        category: getPhonemeCategory(phoneme),
      }))
      .sort((a, b) => a.accuracy - b.accuracy)
  }, [history])

  const categoryData = useMemo(() => {
    return CATEGORIES.map(cat => {
      const catPhonemes = phonemeData.filter(p => p.category === cat)
      if (catPhonemes.length === 0) return null
      const avgAccuracy = catPhonemes.reduce((sum, p) => sum + p.accuracy, 0) / catPhonemes.length
      return {
        category: cat,
        avgAccuracy,
        count: catPhonemes.length,
      }
    }).filter(Boolean)
  }, [phonemeData])

  if (phonemeData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Pronunciation Heatmap</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see your phoneme accuracy heatmap.
          </p>
        </CardContent>
      </Card>
    )
  }

  const worstPhonemes = phonemeData.slice(0, 5)
  const bestPhonemes = phonemeData.filter(p => p.total >= 2).slice(-5).reverse()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Heatmap</span>
          <Badge variant="outline">{phonemeData.length} phonemes</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-7 gap-1">
          {CATEGORIES.map(cat => {
            const catItem = categoryData.find(c => c?.category === cat)
            return (
              <div key={cat} className="text-center">
                <p className="text-[10px] text-muted-foreground mb-1 truncate">{cat}</p>
                {catItem ? (
                  <div
                    className={`h-8 rounded ${getAccuracyColor(catItem.avgAccuracy)}`}
                    title={`${cat}: ${(catItem.avgAccuracy * 100).toFixed(0)}%`}
                  />
                ) : (
                  <div className="h-8 rounded bg-muted/30" />
                )}
              </div>
            )
          })}
        </div>

        <div className="flex items-center gap-2 text-xs text-muted-foreground justify-center">
          <span>Low</span>
          <div className="flex gap-1">
            <div className="w-4 h-3 rounded bg-red-600" />
            <div className="w-4 h-3 rounded bg-red-400" />
            <div className="w-4 h-3 rounded bg-orange-500" />
            <div className="w-4 h-3 rounded bg-yellow-500" />
            <div className="w-4 h-3 rounded bg-green-500" />
          </div>
          <span>High</span>
        </div>

        {worstPhonemes.length > 0 && (
          <div>
            <p className="text-sm font-medium mb-2">Needs Practice</p>
            <div className="space-y-1">
              {worstPhonemes.map(p => (
                <div key={p.phoneme} className="flex items-center gap-2 p-1.5 rounded bg-destructive/5">
                  <Badge variant="outline" className="w-12 justify-center font-mono text-xs">{p.phoneme}</Badge>
                  <span className="text-xs text-muted-foreground w-6">{p.ipa}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div className={`h-full rounded-full ${getAccuracyColor(p.accuracy)}`} style={{ width: `${p.accuracy * 100}%` }} />
                  </div>
                  <span className={`text-xs w-16 text-right ${getAccuracyTextColor(p.accuracy)}`}>
                    {(p.accuracy * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {bestPhonemes.length > 0 && (
          <div>
            <p className="text-sm font-medium mb-2">Mastered</p>
            <div className="space-y-1">
              {bestPhonemes.map(p => (
                <div key={p.phoneme} className="flex items-center gap-2 p-1.5 rounded bg-green-500/5">
                  <Badge variant="outline" className="w-12 justify-center font-mono text-xs">{p.phoneme}</Badge>
                  <span className="text-xs text-muted-foreground w-6">{p.ipa}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div className={`h-full rounded-full ${getAccuracyColor(p.accuracy)}`} style={{ width: `${p.accuracy * 100}%` }} />
                  </div>
                  <span className={`text-xs w-16 text-right ${getAccuracyTextColor(p.accuracy)}`}>
                    {(p.accuracy * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
