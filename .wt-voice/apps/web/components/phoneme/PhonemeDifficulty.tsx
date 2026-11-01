'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { toIPA } from '@/lib/phoneme-controller'

interface PhonemeStats {
  phoneme: string
  ipa: string
  total: number
  matched: number
  mismatched: number
  accuracy: number
}

export default function PhonemeDifficulty() {
  const history = usePhonemeStore(s => s.history)

  const phonemeStats = useMemo(() => {
    if (history.length === 0) return []

    const stats: Record<string, { total: number; matched: number }> = {}

    history.forEach(entry => {
      const maxLen = Math.max(entry.targetPhonemes.length, entry.spokenPhonemes.length)
      for (let i = 0; i < maxLen; i++) {
        const target = entry.targetPhonemes[i]
        if (!target) continue
        if (!stats[target]) stats[target] = { total: 0, matched: 0 }
        stats[target].total++
        if (entry.spokenPhonemes[i] === target) {
          stats[target].matched++
        }
      }
    })

    return Object.entries(stats)
      .map(([phoneme, data]) => ({
        phoneme,
        ipa: toIPA([phoneme])[0] || phoneme,
        total: data.total,
        matched: data.matched,
        mismatched: data.total - data.matched,
        accuracy: data.total > 0 ? data.matched / data.total : 1,
      }))
      .sort((a, b) => a.accuracy - b.accuracy)
  }, [history])

  const hardPhonemes = phonemeStats.filter(p => p.accuracy < 0.7 && p.total >= 3)
  const masteredPhonemes = phonemeStats.filter(p => p.accuracy >= 0.9 && p.total >= 3)

  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Phoneme Difficulty</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see which phonemes you find challenging.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Phoneme Difficulty</span>
          <div className="flex gap-2">
            {hardPhonemes.length > 0 && (
              <Badge variant="destructive">{hardPhonemes.length} challenging</Badge>
            )}
            {masteredPhonemes.length > 0 && (
              <Badge variant="default" className="bg-success">{masteredPhonemes.length} mastered</Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {hardPhonemes.length > 0 && (
          <div>
            <p className="text-sm font-medium text-destructive mb-2">Needs Practice</p>
            <div className="space-y-2">
              {hardPhonemes.slice(0, 5).map(p => (
                <div key={p.phoneme} className="flex items-center gap-3 p-2 rounded bg-destructive/5">
                  <Badge variant="outline" className="w-12 justify-center">{p.phoneme}</Badge>
                  <span className="text-sm text-muted-foreground w-8">{p.ipa}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-destructive"
                      style={{ width: `${p.accuracy * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-16 text-right">
                    {p.matched}/{p.total} ({(p.accuracy * 100).toFixed(0)}%)
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {masteredPhonemes.length > 0 && (
          <Collapsible>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
                <span className="font-medium">Mastered Phonemes</span>
                <Badge variant="default" className="bg-success">{masteredPhonemes.length}</Badge>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-2 flex flex-wrap gap-2">
                {masteredPhonemes.map(p => (
                  <div key={p.phoneme} className="flex items-center gap-1 p-1.5 rounded bg-success/10">
                    <Badge variant="outline" className="text-xs">{p.phoneme}</Badge>
                    <span className="text-xs text-muted-foreground">{p.ipa}</span>
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">All Phoneme Accuracy</span>
              <span className="text-xs text-muted-foreground">{phonemeStats.length} phonemes</span>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-1 max-h-[300px] overflow-y-auto">
              {phonemeStats.map(p => (
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
