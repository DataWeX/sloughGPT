'use client'

import { useMemo, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { toIPA } from '@/lib/phoneme-controller'

interface PhonemeNode {
  phoneme: string
  ipa: string
  count: number
  accuracy: number
  category: string
}

interface PhonemeEdge {
  from: string
  to: string
  count: number
}

const CATEGORY_COLORS: Record<string, string> = {
  Vowels: '#eab308',
  Stops: '#3b82f6',
  Fricatives: '#a855f7',
  Affricates: '#ec4899',
  Nasals: '#22c55e',
  Liquids: '#f97316',
  Glides: '#06b6d4',
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

export default function PhonemeNetwork() {
  const history = usePhonemeStore(s => s.history)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  const { nodes, edges } = useMemo(() => {
    if (history.length === 0) return { nodes: [], edges: [] }

    const phonemeStats: Record<string, { total: number; correct: number }> = {}
    const bigramCounts: Record<string, number> = {}

    history.forEach(entry => {
      const phonemes = entry.targetPhonemes
      phonemes.forEach((p, i) => {
        if (!phonemeStats[p]) phonemeStats[p] = { total: 0, correct: 0 }
        phonemeStats[p].total++
        const score = entry.scores[i] ?? entry.scores[0] ?? 0
        if (score >= 0.8) phonemeStats[p].correct++
      })

      for (let i = 0; i < phonemes.length - 1; i++) {
        const key = `${phonemes[i]}-${phonemes[i + 1]}`
        bigramCounts[key] = (bigramCounts[key] || 0) + 1
      }
    })

    const nodes: PhonemeNode[] = Object.entries(phonemeStats)
      .map(([phoneme, stats]) => ({
        phoneme,
        ipa: toIPA([phoneme])[0] || phoneme,
        count: stats.total,
        accuracy: stats.correct / stats.total,
        category: getPhonemeCategory(phoneme),
      }))
      .filter(n => n.count >= 2)
      .sort((a, b) => b.count - a.count)

    const edges: PhonemeEdge[] = Object.entries(bigramCounts)
      .map(([key, count]) => {
        const [from, to] = key.split('-')
        return { from, to, count }
      })
      .filter(e => e.count >= 2)
      .sort((a, b) => b.count - a.count)
      .slice(0, 20)

    return { nodes, edges }
  }, [history])

  const filteredNodes = selectedCategory
    ? nodes.filter(n => n.category === selectedCategory)
    : nodes

  const filteredEdges = selectedCategory
    ? edges.filter(e =>
        filteredNodes.some(n => n.phoneme === e.from) &&
        filteredNodes.some(n => n.phoneme === e.to)
      )
    : edges

  if (nodes.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Phoneme Network</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Practice more to see phoneme relationships.
          </p>
        </CardContent>
      </Card>
    )
  }

  const categories = [...new Set(nodes.map(n => n.category))]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Phoneme Network</span>
          <Badge variant="outline">{nodes.length} nodes, {edges.length} connections</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge
            variant={!selectedCategory ? 'default' : 'outline'}
            className="cursor-pointer"
            onClick={() => setSelectedCategory(null)}
          >
            All
          </Badge>
          {categories.map(cat => (
            <Badge
              key={cat}
              variant={selectedCategory === cat ? 'default' : 'outline'}
              className="cursor-pointer"
              onClick={() => setSelectedCategory(cat)}
              style={selectedCategory === cat ? { backgroundColor: CATEGORY_COLORS[cat] } : {}}
            >
              {cat}
            </Badge>
          ))}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {filteredNodes.slice(0, 12).map(node => (
            <div
              key={node.phoneme}
              className="p-2 rounded-lg border text-center"
              style={{ borderColor: CATEGORY_COLORS[node.category] || '#888' }}
            >
              <p className="font-mono text-sm font-bold">{node.phoneme}</p>
              <p className="text-xs text-muted-foreground">{node.ipa}</p>
              <div className="h-1.5 rounded-full bg-muted mt-1 overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${node.accuracy * 100}%`,
                    backgroundColor: CATEGORY_COLORS[node.category] || '#888',
                  }}
                />
              </div>
              <p className="text-[10px] text-muted-foreground mt-0.5">
                {(node.accuracy * 100).toFixed(0)}% ({node.count}x)
              </p>
            </div>
          ))}
        </div>

        {filteredEdges.length > 0 && (
          <div>
            <p className="text-sm font-medium mb-2">Strongest Connections</p>
            <div className="space-y-1">
              {filteredEdges.slice(0, 8).map((edge, i) => (
                <div key={`${edge.from}-${edge.to}`} className="flex items-center gap-2 text-xs">
                  <Badge variant="outline" className="font-mono w-10 justify-center">{edge.from}</Badge>
                  <span className="text-muted-foreground">→</span>
                  <Badge variant="outline" className="font-mono w-10 justify-center">{edge.to}</Badge>
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${(edge.count / filteredEdges[0].count) * 100}%` }}
                    />
                  </div>
                  <span className="text-muted-foreground w-8 text-right">{edge.count}x</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
