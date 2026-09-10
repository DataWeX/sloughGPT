'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { toIPA } from '@/lib/phoneme-controller'

interface PhonemeFrequency {
  phoneme: string
  ipa: string
  count: number
  percentage: number
  category: string
}

function getPhonemeCategory(phoneme: string): string {
  const stops = ['P', 'B', 'T', 'D', 'K', 'G']
  const fricatives = ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH']
  const affricates = ['CH', 'JH']
  const nasals = ['M', 'N', 'NG']
  const liquids = ['L', 'R']
  const glides = ['W', 'Y']
  const vowels = ['IY', 'IH', 'EY', 'EH', 'AE', 'AA', 'AH', 'AO', 'OW', 'OY', 'UH', 'UW', 'ER', 'AX']

  if (stops.includes(phoneme)) return 'Stops'
  if (fricatives.includes(phoneme)) return 'Fricatives'
  if (affricates.includes(phoneme)) return 'Affricates'
  if (nasals.includes(phoneme)) return 'Nasals'
  if (liquids.includes(phoneme)) return 'Liquids'
  if (glides.includes(phoneme)) return 'Glides'
  if (vowels.includes(phoneme)) return 'Vowels'
  return 'Other'
}

function getCategoryColor(category: string): string {
  switch (category) {
    case 'Stops': return 'bg-blue-100 text-blue-800'
    case 'Fricatives': return 'bg-purple-100 text-purple-800'
    case 'Affricates': return 'bg-pink-100 text-pink-800'
    case 'Nasals': return 'bg-green-100 text-green-800'
    case 'Liquids': return 'bg-orange-100 text-orange-800'
    case 'Glides': return 'bg-cyan-100 text-cyan-800'
    case 'Vowels': return 'bg-yellow-100 text-yellow-800'
    default: return 'bg-gray-100 text-gray-800'
  }
}

export default function PhonemeFrequency() {
  const history = usePhonemeStore(s => s.history)

  const frequencies = useMemo(() => {
    if (history.length === 0) return { phonemes: [], categories: [], totalPhonemes: 0 }

    const phonemeCounts: Record<string, number> = {}
    let totalPhonemes = 0

    history.forEach(entry => {
      entry.targetPhonemes.forEach(phoneme => {
        phonemeCounts[phoneme] = (phonemeCounts[phoneme] || 0) + 1
        totalPhonemes++
      })
    })

    const phonemes: PhonemeFrequency[] = Object.entries(phonemeCounts)
      .map(([phoneme, count]) => ({
        phoneme,
        ipa: toIPA([phoneme])[0] || phoneme,
        count,
        percentage: (count / totalPhonemes) * 100,
        category: getPhonemeCategory(phoneme),
      }))
      .sort((a, b) => b.count - a.count)

    const categoryCounts: Record<string, number> = {}
    phonemes.forEach(p => {
      categoryCounts[p.category] = (categoryCounts[p.category] || 0) + p.count
    })

    const categories = Object.entries(categoryCounts)
      .map(([name, count]) => ({
        name,
        count,
        percentage: (count / totalPhonemes) * 100,
      }))
      .sort((a, b) => b.count - a.count)

    return { phonemes, categories, totalPhonemes }
  }, [history])

  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Phoneme Frequency</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see phoneme frequency analysis.
          </p>
        </CardContent>
      </Card>
    )
  }

  const topPhonemes = frequencies.phonemes.slice(0, 10)
  const bottomPhonemes = frequencies.phonemes.slice(-5).reverse()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Phoneme Frequency</span>
          <Badge variant="outline">{frequencies.totalPhonemes} total</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
          {frequencies.categories.slice(0, 4).map(cat => (
            <div key={cat.name} className="p-2 rounded-lg bg-muted/30">
              <p className="text-lg font-bold">{cat.count}</p>
              <p className="text-xs text-muted-foreground">{cat.name}</p>
            </div>
          ))}
        </div>

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Most Common Phonemes</span>
              <Badge variant="default">Top 10</Badge>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-2">
              {topPhonemes.map((p, i) => (
                <div key={p.phoneme} className="flex items-center gap-3 p-2 rounded bg-muted/20">
                  <span className="text-xs text-muted-foreground w-4">{i + 1}</span>
                  <Badge variant="outline" className="w-12 justify-center">{p.phoneme}</Badge>
                  <span className="text-xs text-muted-foreground w-8">{p.ipa}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${p.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-16 text-right">
                    {p.count} ({p.percentage.toFixed(1)}%)
                  </span>
                  <Badge className={`text-xs ${getCategoryColor(p.category)}`}>
                    {p.category}
                  </Badge>
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Least Common Phonemes</span>
              <Badge variant="secondary">Bottom 5</Badge>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-2">
              {bottomPhonemes.map((p, i) => (
                <div key={p.phoneme} className="flex items-center gap-3 p-2 rounded bg-muted/20">
                  <Badge variant="outline" className="w-12 justify-center">{p.phoneme}</Badge>
                  <span className="text-xs text-muted-foreground w-8">{p.ipa}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-destructive"
                      style={{ width: `${p.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-16 text-right">
                    {p.count} ({p.percentage.toFixed(1)}%)
                  </span>
                  <Badge className={`text-xs ${getCategoryColor(p.category)}`}>
                    {p.category}
                  </Badge>
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Category Breakdown</span>
              <span className="text-xs text-muted-foreground">{frequencies.categories.length} categories</span>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-2">
              {frequencies.categories.map(cat => (
                <div key={cat.name} className="flex items-center gap-3">
                  <span className="text-sm w-24">{cat.name}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${cat.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-16 text-right">
                    {cat.count} ({cat.percentage.toFixed(1)}%)
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
