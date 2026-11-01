'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { toIPA, PHONEME_LANGUAGES } from '@/lib/phoneme-controller'

interface PhonemePattern {
  phoneme: string
  ipa: string
  total: number
  substitutedWith: Record<string, number>
  accuracy: number
}

interface WordPattern {
  word: string
  lang: string
  count: number
  avgScore: number
  commonMistakes: string[]
}

export default function PronunciationPatterns() {
  const history = usePhonemeStore(s => s.history)

  const patterns = useMemo(() => {
    if (history.length === 0) return { phonemePatterns: [], wordPatterns: [], commonSubstitutions: [] }

    const phonemeStats: Record<string, { total: number; matched: number; substitutions: Record<string, number> }> = {}
    const wordStats: Record<string, { count: number; totalScore: number; mistakes: string[] }> = {}

    history.forEach(entry => {
      const maxLen = Math.max(entry.targetPhonemes.length, entry.spokenPhonemes.length)
      for (let i = 0; i < maxLen; i++) {
        const target = entry.targetPhonemes[i]
        const spoken = entry.spokenPhonemes[i]
        if (!target) continue
        if (!phonemeStats[target]) phonemeStats[target] = { total: 0, matched: 0, substitutions: {} }
        phonemeStats[target].total++
        if (spoken === target) {
          phonemeStats[target].matched++
        } else if (spoken) {
          phonemeStats[target].substitutions[spoken] = (phonemeStats[target].substitutions[spoken] || 0) + 1
        }
      }

      const wordKey = `${entry.language}:${entry.target}`
      if (!wordStats[wordKey]) wordStats[wordKey] = { count: 0, totalScore: 0, mistakes: [] }
      wordStats[wordKey].count++
      wordStats[wordKey].totalScore += entry.score
      if (entry.score < 0.7) {
        const mismatches = entry.targetPhonemes
          .filter((p, i) => entry.spokenPhonemes[i] && entry.spokenPhonemes[i] !== p)
          .map(p => p)
        wordStats[wordKey].mistakes.push(...mismatches)
      }
    })

    const phonemePatterns: PhonemePattern[] = Object.entries(phonemeStats)
      .map(([phoneme, data]) => ({
        phoneme,
        ipa: toIPA([phoneme])[0] || phoneme,
        total: data.total,
        substitutedWith: data.substitutions,
        accuracy: data.total > 0 ? data.matched / data.total : 1,
      }))
      .sort((a, b) => a.accuracy - b.accuracy)

    const wordPatterns: WordPattern[] = Object.entries(wordStats)
      .map(([key, data]) => {
        const [lang, ...wordParts] = key.split(':')
        const word = wordParts.join(':')
        const mistakeCounts: Record<string, number> = {}
        data.mistakes.forEach(m => { mistakeCounts[m] = (mistakeCounts[m] || 0) + 1 })
        const commonMistakes = Object.entries(mistakeCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([m]) => m)
        return {
          word,
          lang,
          count: data.count,
          avgScore: data.totalScore / data.count,
          commonMistakes,
        }
      })
      .sort((a, b) => a.avgScore - b.avgScore)

    const allSubstitutions: Record<string, { from: string; to: string; count: number }> = {}
    phonemePatterns.forEach(p => {
      Object.entries(p.substitutedWith).forEach(([to, count]) => {
        const key = `${p.phoneme}->${to}`
        allSubstitutions[key] = { from: p.phoneme, to, count }
      })
    })
    const commonSubstitutions = Object.values(allSubstitutions)
      .sort((a, b) => b.count - a.count)
      .slice(0, 10)

    return { phonemePatterns, wordPatterns, commonSubstitutions }
  }, [history])

  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Pronunciation Patterns</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see your pronunciation patterns.
          </p>
        </CardContent>
      </Card>
    )
  }

  const hardPhonemes = patterns.phonemePatterns.filter(p => p.accuracy < 0.7 && p.total >= 3)
  const hardWords = patterns.wordPatterns.filter(w => w.avgScore < 0.7 && w.count >= 2)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Patterns</span>
          <div className="flex gap-2">
            {hardPhonemes.length > 0 && (
              <Badge variant="destructive">{hardPhonemes.length} hard phonemes</Badge>
            )}
            {hardWords.length > 0 && (
              <Badge variant="secondary">{hardWords.length} hard words</Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {patterns.commonSubstitutions.length > 0 && (
          <div>
            <p className="text-sm font-medium text-muted-foreground mb-2">Common Substitutions</p>
            <div className="space-y-1">
              {patterns.commonSubstitutions.map((sub, i) => (
                <div key={i} className="flex items-center gap-2 p-2 rounded bg-muted/30 text-sm">
                  <Badge variant="outline">{sub.from}</Badge>
                  <span className="text-muted-foreground">→</span>
                  <Badge variant="outline">{sub.to}</Badge>
                  <span className="text-xs text-muted-foreground ml-auto">
                    {sub.count}x ({toIPA([sub.from])[0]} → {toIPA([sub.to])[0]})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {hardPhonemes.length > 0 && (
          <Collapsible>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
                <span className="font-medium">Challenging Phonemes</span>
                <Badge variant="destructive">{hardPhonemes.length}</Badge>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-2 space-y-2">
                {hardPhonemes.slice(0, 5).map(p => (
                  <div key={p.phoneme} className="p-2 rounded bg-destructive/5">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="outline">{p.phoneme}</Badge>
                      <span className="text-xs text-muted-foreground">{p.ipa}</span>
                      <span className="text-xs text-muted-foreground ml-auto">
                        {(p.accuracy * 100).toFixed(0)}% accuracy
                      </span>
                    </div>
                    {Object.keys(p.substitutedWith).length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        <span className="text-xs text-muted-foreground">Often replaced with:</span>
                        {Object.entries(p.substitutedWith)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 3)
                          .map(([sub, count]) => (
                            <Badge key={sub} variant="outline" className="text-xs">
                              {sub} ({count}x)
                            </Badge>
                          ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {hardWords.length > 0 && (
          <Collapsible>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
                <span className="font-medium">Words to Practice</span>
                <Badge variant="secondary">{hardWords.length}</Badge>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-2 space-y-2">
                {hardWords.slice(0, 5).map(w => (
                  <div key={`${w.lang}:${w.word}`} className="p-2 rounded bg-secondary/5">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium">{w.word}</span>
                      <Badge variant="outline" className="text-xs">
                        {PHONEME_LANGUAGES.find(l => l.value === w.lang)?.label}
                      </Badge>
                      <span className="text-xs text-muted-foreground ml-auto">
                        {(w.avgScore * 100).toFixed(0)}% avg
                      </span>
                    </div>
                    {w.commonMistakes.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        <span className="text-xs text-muted-foreground">Struggles with:</span>
                        {w.commonMistakes.map((m, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {m} ({toIPA([m])[0]})
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-300 cursor-pointer text-sm">
              <span className="font-medium">All Phoneme Accuracy</span>
              <span className="text-xs text-muted-foreground">{patterns.phonemePatterns.length} phonemes</span>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-1 max-h-[300px] overflow-y-auto">
              {patterns.phonemePatterns.slice(0, 15).map(p => (
                <div key={p.phoneme} className="flex items-center gap-2 text-xs">
                  <Badge variant="outline" className="w-10 justify-center">{p.phoneme}</Badge>
                  <span className="text-muted-foreground w-6">{p.ipa}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        p.accuracy >= 0.9 ? 'bg-success' :
                        p.accuracy >= 0.7 ? 'bg-primary' :
                        'bg-destructive'
                      }`}
                      style={{ width: `${p.accuracy * 100}%` }}
                    />
                  </div>
                  <span className="w-12 text-right text-muted-foreground">
                    {(p.accuracy * 100).toFixed(0)}%
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
