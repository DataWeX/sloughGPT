'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { toIPA } from '@/lib/phoneme-controller'

interface WordFamily {
  ending: string
  ipa: string
  words: string[]
  count: number
}

function groupByEnding(history: NonNullable<ReturnType<typeof usePhonemeStore.getState>['history']>): WordFamily[] {
  const familyMap: Record<string, { phonemes: string[]; words: Set<string> }> = {}

  history.forEach(entry => {
    const phonemes = entry.targetPhonemes
    if (phonemes.length < 2) return

    for (let endLen = 1; endLen <= Math.min(3, phonemes.length); endLen++) {
      const ending = phonemes.slice(-endLen).join('-')
      if (!familyMap[ending]) {
        familyMap[ending] = { phonemes: phonemes.slice(-endLen), words: new Set() }
      }
      familyMap[ending].words.add(entry.targetWord)
    }
  })

  return Object.entries(familyMap)
    .filter(([, v]) => v.words.size >= 2)
    .map(([ending, v]) => ({
      ending,
      ipa: toIPA(v.phonemes).join(''),
      words: Array.from(v.words).sort(),
      count: v.words.size,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 15)
}

export default function WordFamilies() {
  const history = usePhonemeStore(s => s.history)

  const families = useMemo(() => groupByEnding(history), [history])

  if (history.length === 0 || families.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Word Families</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more words to see groups by phoneme endings.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Word Families</span>
          <Badge variant="outline">{families.length} endings</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Words you practice grouped by their phoneme endings.
        </p>
        {families.map(family => (
          <Collapsible key={family.ending}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/30 transition-colors text-sm cursor-pointer">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="min-w-[60px] justify-center font-mono">
                    {family.ending}
                  </Badge>
                  <span className="text-xs text-muted-foreground">{family.ipa}</span>
                </div>
                <Badge variant="secondary">{family.count} words</Badge>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-2 pt-0 flex flex-wrap gap-2">
                {family.words.map(word => (
                  <Badge key={word} variant="outline" className="text-xs">{word}</Badge>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  )
}
