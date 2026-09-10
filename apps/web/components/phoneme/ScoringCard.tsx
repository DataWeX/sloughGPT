'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge, Kbd } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Progress } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeScoreResult, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

export default function ScoringCard() {
  const [target, setTarget] = useState('hello')
  const [spoken, setSpoken] = useState('hello')
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PhonemeScoreResult | null>(null)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const doScore = useCallback(async (t: string, s: string, l: PhonemeLanguage) => {
    if (!t.trim() || !s.trim()) { setResult(null); return }
    setLoading(true)
    try {
      const res = await phonemeController.score(t, s, l)
      setResult(res)
      addToHistory({
        target: res.target,
        spoken: res.spoken,
        score: res.score,
        language: res.language,
        targetPhonemes: res.target_phonemes,
        spokenPhonemes: res.spoken_phonemes,
      })
    } catch {
      // silent — debounce will retry
    } finally {
      setLoading(false)
    }
  }, [addToHistory])

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doScore(target, spoken, language), 400)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [target, spoken, language, doScore])

  const handleScore = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    doScore(target, spoken, language)
  }, [target, spoken, language, doScore])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Score Pronunciation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <Input
            value={target}
            onChange={e => setTarget(e.target.value)}
            placeholder="Target word..."
            className="flex-1"
          />
          <Input
            value={spoken}
            onChange={e => setSpoken(e.target.value)}
            placeholder="Your attempt..."
            className="flex-1"
            onKeyDown={e => e.key === 'Enter' && handleScore()}
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
          <Button onClick={handleScore} disabled={loading || !target.trim() || !spoken.trim()}>
            {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
            Score
            <Kbd className="ml-2 hidden sm:inline-flex">Enter</Kbd>
          </Button>
        </div>

        {result && (
          <div className="space-y-4 p-4 rounded-lg bg-muted/50 animate-in fade-in slide-in-from-top-1 duration-200">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold">{(result.score * 100).toFixed(0)}%</p>
                <p className="text-xs text-muted-foreground">Score</p>
              </div>
              <div>
                <p className="text-2xl font-bold">{(result.precision * 100).toFixed(0)}%</p>
                <p className="text-xs text-muted-foreground">Precision</p>
              </div>
              <div>
                <p className="text-2xl font-bold">{(result.recall * 100).toFixed(0)}%</p>
                <p className="text-xs text-muted-foreground">Recall</p>
              </div>
            </div>

            <Progress value={result.score * 100} />

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Target Phonemes:</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {result.target_phonemes.map((p, i) => (
                    <Badge key={i} variant="outline">{p}</Badge>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">Spoken Phonemes:</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {result.spoken_phonemes.map((p, i) => (
                    <Badge key={i} variant={result.target_phonemes[i] === p ? 'default' : 'destructive'}>{p}</Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
