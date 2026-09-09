'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Progress } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeEncodeResult, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'

export default function ComparisonCard() {
  const [word1, setWord1] = useState('hello')
  const [word2, setWord2] = useState('world')
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [loading, setLoading] = useState(false)
  const [result1, setResult1] = useState<PhonemeEncodeResult | null>(null)
  const [result2, setResult2] = useState<PhonemeEncodeResult | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleCompare = useCallback(async () => {
    if (!word1.trim() || !word2.trim()) return
    setLoading(true)
    try {
      const [r1, r2] = await Promise.all([
        phonemeController.encode(word1, language),
        phonemeController.encode(word2, language),
      ])
      setResult1(r1)
      setResult2(r2)
    } catch (err) {
      addToast('Comparison failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [word1, word2, language, addToast])

  const common = result1 && result2
    ? result1.phonemes.filter(p => result2.phonemes.includes(p))
    : []
  const allPhonemes = result1 && result2
    ? [...new Set([...result1.phonemes, ...result2.phonemes])]
    : []
  const similarity = allPhonemes.length > 0 ? (common.length / allPhonemes.length) * 100 : 0

  return (
    <Card>
      <CardHeader>
        <CardTitle>Compare Words</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-3">
          <Input
            value={word1}
            onChange={e => setWord1(e.target.value)}
            placeholder="Word 1..."
            className="flex-1"
          />
          <Input
            value={word2}
            onChange={e => setWord2(e.target.value)}
            placeholder="Word 2..."
            className="flex-1"
            onKeyDown={e => e.key === 'Enter' && handleCompare()}
          />
          <Select value={language} onValueChange={v => setLanguage(v as PhonemeLanguage)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PHONEME_LANGUAGES.map(l => (
                <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={handleCompare} disabled={loading || !word1.trim() || !word2.trim()}>
            {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
            Compare
          </Button>
        </div>

        {result1 && result2 && (
          <div className="space-y-4 p-4 rounded-lg bg-muted/50">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-sm font-medium">{word1}</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {result1.phonemes.map((p, i) => (
                    <Badge key={i} variant="outline">{p}</Badge>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-sm font-medium">{word2}</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {result2.phonemes.map((p, i) => (
                    <Badge key={i} variant="outline">{p}</Badge>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Similarity</span>
                <span className="text-sm font-bold">{similarity.toFixed(1)}%</span>
              </div>
              <Progress value={similarity} />
            </div>

            {common.length > 0 && (
              <div>
                <p className="text-sm font-medium text-muted-foreground">Common Phonemes:</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {common.map((p, i) => (
                    <Badge key={i} variant="default">{p}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
