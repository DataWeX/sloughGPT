'use client'

import { useMemo, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, Input } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { toIPA } from '@/lib/phoneme-controller'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import PhonemeSkeleton from './PhonemeSkeleton'

interface ComparisonResult {
  word1: string
  word2: string
  phonemes1: string[]
  phonemes2: string[]
  ipa1: string
  ipa2: string
  matchPositions: number[]
  diffPositions: number[]
  similarity: number
}

export default function PronunciationComparison() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [word1, setWord1] = useState('')
  const [word2, setWord2] = useState('')
  const [result, setResult] = useState<ComparisonResult | null>(null)
  const [loading, setLoading] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const compare = async () => {
    if (!word1.trim() || !word2.trim()) return
    setLoading(true)
    try {
      const [r1, r2] = await Promise.all([
        phonemeController.encode(word1.trim().toLowerCase(), language),
        phonemeController.encode(word2.trim().toLowerCase(), language),
      ])

      const phonemes1 = r1.phonemes
      const phonemes2 = r2.phonemes
      const ipa1 = toIPA(phonemes1).join('')
      const ipa2 = toIPA(phonemes2).join('')

      const maxLen = Math.max(phonemes1.length, phonemes2.length)
      const matchPositions: number[] = []
      const diffPositions: number[] = []

      for (let i = 0; i < maxLen; i++) {
        if (phonemes1[i] === phonemes2[i]) {
          matchPositions.push(i)
        } else {
          diffPositions.push(i)
        }
      }

      const similarity = matchPositions.length / maxLen

      setResult({
        word1: word1.trim().toLowerCase(),
        word2: word2.trim().toLowerCase(),
        phonemes1,
        phonemes2,
        ipa1,
        ipa2,
        matchPositions,
        diffPositions,
        similarity,
      })
    } catch {
      addToast({ type: 'error', message: 'Failed to encode words' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Comparison</span>
          {result && (
            <Badge variant={result.similarity >= 0.8 ? 'default' : 'outline'}>
              {(result.similarity * 100).toFixed(0)}% match
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <Select value={language} onValueChange={v => setLanguage(v as PhonemeLanguage)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PHONEME_LANGUAGES.map(lang => (
                <SelectItem key={lang.value} value={lang.value}>{lang.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Input
            placeholder="First word..."
            value={word1}
            onChange={e => setWord1(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && compare()}
          />
          <Input
            placeholder="Second word..."
            value={word2}
            onChange={e => setWord2(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && compare()}
          />
        </div>

        <Button onClick={compare} disabled={!word1.trim() || !word2.trim() || loading} className="w-full">
          {loading ? 'Comparing...' : 'Compare'}
        </Button>

        {result && (
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-muted/30">
              <p className="text-xs text-muted-foreground mb-2">Phoneme comparison:</p>
              <div className="flex flex-wrap gap-1">
                {Array.from({ length: Math.max(result.phonemes1.length, result.phonemes2.length) }).map((_, i) => {
                  const p1 = result.phonemes1[i] || '-'
                  const p2 = result.phonemes2[i] || '-'
                  const isMatch = result.matchPositions.includes(i)
                  return (
                    <div
                      key={i}
                      className={`flex flex-col items-center p-1 rounded ${
                        isMatch ? 'bg-green-500/10' : 'bg-destructive/10'
                      }`}
                    >
                      <span className="text-xs font-mono">{p1}</span>
                      <span className="text-[10px] text-muted-foreground">{isMatch ? '=' : '≠'}</span>
                      <span className="text-xs font-mono">{p2}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-muted/20">
                <p className="text-xs text-muted-foreground mb-1">{result.word1}</p>
                <p className="text-sm font-mono">{result.ipa1}</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/20">
                <p className="text-xs text-muted-foreground mb-1">{result.word2}</p>
                <p className="text-sm font-mono">{result.ipa2}</p>
              </div>
            </div>

            <div className="flex justify-center gap-4 text-xs text-muted-foreground">
              <span className="text-green-600">{result.matchPositions.length} matches</span>
              <span className="text-destructive">{result.diffPositions.length} differences</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
