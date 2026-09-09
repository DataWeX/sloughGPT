'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge, Kbd } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeEncodeResult, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'

export default function EncodingCard() {
  const [text, setText] = useState('hello world')
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PhonemeEncodeResult | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleEncode = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await phonemeController.encode(text, language)
      setResult(res)
    } catch (err) {
      addToast('Encoding failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [text, language, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Encode Text to Phonemes</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-3">
          <Input
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Enter text to encode..."
            onKeyDown={e => e.key === 'Enter' && handleEncode()}
            className="flex-1"
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
          <Button onClick={handleEncode} disabled={loading || !text.trim()}>
            {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
            Encode
            <Kbd className="ml-2 hidden sm:inline-flex">Enter</Kbd>
          </Button>
        </div>

        {result && (
          <div className="space-y-3 p-4 rounded-lg bg-muted/50">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-muted-foreground">Language:</span>
              <Badge variant="secondary">{result.language.toUpperCase()}</Badge>
            </div>

            <div>
              <span className="text-sm font-medium text-muted-foreground">Phonemes:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {result.phonemes.map((p, i) => (
                  <Badge key={i} variant="outline">{p}</Badge>
                ))}
              </div>
            </div>

            <div>
              <span className="text-sm font-medium text-muted-foreground">IDs:</span>
              <code className="block mt-1 text-xs p-2 rounded bg-background font-mono">
                [{result.ids.join(', ')}]
              </code>
            </div>

            <div>
              <span className="text-sm font-medium text-muted-foreground">Decoded:</span>
              <p className="mt-1 text-sm">{result.decoded}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
