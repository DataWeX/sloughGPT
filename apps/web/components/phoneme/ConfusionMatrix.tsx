'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'

interface Confusion {
  target: string
  spoken: string
  count: number
}

export default function ConfusionMatrix() {
  const history = usePhonemeStore(s => s.history)

  const confusions = useMemo(() => {
    if (history.length === 0) return []

    const confusionMap: Record<string, number> = {}

    history.forEach(entry => {
      entry.targetPhonemes.forEach((target, i) => {
        const spoken = entry.spokenPhonemes?.[i] || target
        if (target !== spoken) {
          const key = `${target}→${spoken}`
          confusionMap[key] = (confusionMap[key] || 0) + 1
        }
      })
    })

    return Object.entries(confusionMap)
      .map(([key, count]) => {
        const [target, spoken] = key.split('→')
        return { target, spoken, count }
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, 15)
  }, [history])

  if (confusions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Confusion Patterns</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see which phonemes you confuse.
          </p>
        </CardContent>
      </Card>
    )
  }

  const maxCount = confusions[0]?.count || 1

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Confusion Patterns</span>
          <Badge variant="outline">{confusions.length} patterns</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Common phoneme substitutions in your practice.
        </p>

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Most Common Confusions</span>
              <Badge variant="default">Top {confusions.length}</Badge>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-2">
              {confusions.map((c, i) => (
                <div key={`${c.target}-${c.spoken}`} className="flex items-center gap-3 p-2 rounded bg-muted/20">
                  <span className="text-xs text-muted-foreground w-4">{i + 1}</span>
                  <div className="flex items-center gap-1 min-w-[80px]">
                    <Badge variant="outline" className="font-mono text-xs">{c.target}</Badge>
                    <span className="text-xs text-muted-foreground">→</span>
                    <Badge variant="destructive" className="font-mono text-xs">{c.spoken}</Badge>
                  </div>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-destructive"
                      style={{ width: `${(c.count / maxCount) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-12 text-right">{c.count}x</span>
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="p-3 rounded-lg bg-muted/30 text-sm">
          <p className="font-medium mb-1">Tip</p>
          <p className="text-muted-foreground">
            {confusions.length > 0 && (
              <>
                You most often confuse <strong>{confusions[0].target}</strong> with <strong>{confusions[0].spoken}</strong>.
                Focus on the minimal pairs between these sounds.
              </>
            )}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
