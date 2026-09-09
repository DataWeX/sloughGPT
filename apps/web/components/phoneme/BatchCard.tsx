'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeEncodeResult, type PhonemeScoreResult, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'

export default function BatchCard() {
  const [mode, setMode] = useState<'encode' | 'score'>('encode')
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [inputText, setInputText] = useState('hello\nworld\nplease')
  const [scoreInput, setScoreInput] = useState('hello→hello\nworld→worl\nplease→pleas')
  const [loading, setLoading] = useState(false)
  const [encodeResults, setEncodeResults] = useState<PhonemeEncodeResult[]>([])
  const [scoreResults, setScoreResults] = useState<PhonemeScoreResult[]>([])
  const addToast = useToastStore(s => s.addToast)

  const handleBatchEncode = useCallback(async () => {
    const texts = inputText.split('\n').map(s => s.trim()).filter(Boolean)
    if (texts.length === 0) return
    setLoading(true)
    try {
      const res = await phonemeController.batchEncode(texts, language)
      setEncodeResults(res.results)
    } catch {
      addToast('Batch encoding failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [inputText, language, addToast])

  const handleBatchScore = useCallback(async () => {
    const pairs = scoreInput.split('\n')
      .map(s => s.trim())
      .filter(Boolean)
      .map(s => {
        const [target, spoken] = s.split('→').map(x => x?.trim())
        return target && spoken ? { target, spoken } : null
      })
      .filter(Boolean) as { target: string; spoken: string }[]

    if (pairs.length === 0) return
    setLoading(true)
    try {
      const res = await phonemeController.batchScore(pairs, language)
      setScoreResults(res.results)
    } catch {
      addToast('Batch scoring failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [scoreInput, language, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Batch Operations</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-3">
          <Select value={mode} onValueChange={v => setMode(v as 'encode' | 'score')}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="encode">Batch Encode</SelectItem>
              <SelectItem value="score">Batch Score</SelectItem>
            </SelectContent>
          </Select>
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
        </div>

        {mode === 'encode' ? (
          <>
            <Textarea
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              placeholder="Enter one word per line..."
              rows={5}
            />
            <Button onClick={handleBatchEncode} disabled={loading || !inputText.trim()}>
              {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
              Encode All
            </Button>

            {encodeResults.length > 0 && (
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {encodeResults.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-muted/50 text-sm">
                    <span className="font-medium min-w-[80px]">{r.text}</span>
                    <div className="flex flex-wrap gap-1">
                      {r.phonemes.map((p, j) => (
                        <Badge key={j} variant="outline" className="text-xs">{p}</Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            <Textarea
              value={scoreInput}
              onChange={e => setScoreInput(e.target.value)}
              placeholder="Enter target→spoken per line (e.g. hello→helo)"
              rows={5}
            />
            <Button onClick={handleBatchScore} disabled={loading || !scoreInput.trim()}>
              {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
              Score All
            </Button>

            {scoreResults.length > 0 && (
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {scoreResults.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-muted/50 text-sm">
                    <span className="font-medium min-w-[80px]">{r.target}</span>
                    <span className="text-muted-foreground">→</span>
                    <span>{r.spoken}</span>
                    <Badge
                      variant={r.score >= 0.8 ? 'default' : r.score >= 0.5 ? 'secondary' : 'destructive'}
                      className="ml-auto"
                    >
                      {(r.score * 100).toFixed(0)}%
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
