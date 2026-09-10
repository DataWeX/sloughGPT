'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { toIPA } from '@/lib/phoneme-controller'

interface Cluster {
  phonemes: string[]
  ipa: string
  count: number
  percentage: number
}

function extractClusters(phonemes: string[], size: number): string[][] {
  const clusters: string[][] = []
  for (let i = 0; i <= phonemes.length - size; i++) {
    clusters.push(phonemes.slice(i, i + size))
  }
  return clusters
}

export default function PhonemeClusters() {
  const history = usePhonemeStore(s => s.history)

  const { bigrams, trigrams } = useMemo(() => {
    if (history.length === 0) return { bigrams: [], trigrams: [] }

    const bigramCounts: Record<string, { phonemes: string[]; count: number }> = {}
    const trigramCounts: Record<string, { phonemes: string[]; count: number }> = {}

    history.forEach(entry => {
      const phonemes = entry.targetPhonemes

      extractClusters(phonemes, 2).forEach(cluster => {
        const key = cluster.join('-')
        bigramCounts[key] = bigramCounts[key] || { phonemes: cluster, count: 0 }
        bigramCounts[key].count++
      })

      extractClusters(phonemes, 3).forEach(cluster => {
        const key = cluster.join('-')
        trigramCounts[key] = trigramCounts[key] || { phonemes: cluster, count: 0 }
        trigramCounts[key].count++
      })
    })

    const totalBigrams = Object.values(bigramCounts).reduce((sum, b) => sum + b.count, 0)
    const totalTrigrams = Object.values(trigramCounts).reduce((sum, t) => sum + t.count, 0)

    const bigrams: Cluster[] = Object.values(bigramCounts)
      .map(b => ({
        phonemes: b.phonemes,
        ipa: toIPA(b.phonemes).join(''),
        count: b.count,
        percentage: (b.count / totalBigrams) * 100,
      }))
      .sort((a, b) => b.count - a.count)

    const trigrams: Cluster[] = Object.values(trigramCounts)
      .map(t => ({
        phonemes: t.phonemes,
        ipa: toIPA(t.phonemes).join(''),
        count: t.count,
        percentage: (t.count / totalTrigrams) * 100,
      }))
      .sort((a, b) => b.count - a.count)

    return { bigrams: bigrams.slice(0, 20), trigrams: trigrams.slice(0, 15) }
  }, [history])

  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Phoneme Clusters</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see common phoneme combinations.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Phoneme Clusters</span>
          <Badge variant="outline">Common Combinations</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Bigrams (2-phoneme combinations)</span>
              <Badge variant="default">Top 20</Badge>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-2">
              {bigrams.map((cluster, i) => (
                <div key={cluster.phonemes.join('-')} className="flex items-center gap-3 p-2 rounded bg-muted/20">
                  <span className="text-xs text-muted-foreground w-4">{i + 1}</span>
                  <Badge variant="outline" className="min-w-[60px] justify-center">
                    {cluster.phonemes.join('-')}
                  </Badge>
                  <span className="text-xs text-muted-foreground w-12">{cluster.ipa}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${cluster.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-16 text-right">
                    {cluster.count} ({cluster.percentage.toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Trigrams (3-phoneme combinations)</span>
              <Badge variant="secondary">Top 15</Badge>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-2">
              {trigrams.map((cluster, i) => (
                <div key={cluster.phonemes.join('-')} className="flex items-center gap-3 p-2 rounded bg-muted/20">
                  <span className="text-xs text-muted-foreground w-4">{i + 1}</span>
                  <Badge variant="outline" className="min-w-[80px] justify-center">
                    {cluster.phonemes.join('-')}
                  </Badge>
                  <span className="text-xs text-muted-foreground w-16">{cluster.ipa}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{ width: `${cluster.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-16 text-right">
                    {cluster.count} ({cluster.percentage.toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="p-3 rounded-lg bg-muted/30 text-sm">
          <p className="font-medium mb-1">Insight</p>
          <p className="text-muted-foreground">
            {bigrams.length > 0 && (
              <>
                Your most common bigram is <strong>{bigrams[0].phonemes.join('-')}</strong> ({bigrams[0].ipa}),
                appearing in {bigrams[0].percentage.toFixed(1)}% of your practice.
              </>
            )}
            {trigrams.length > 0 && (
              <> The most common trigram is <strong>{trigrams[0].phonemes.join('-')}</strong> ({trigrams[0].ipa}).</>
            )}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
