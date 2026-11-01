'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'
import PhonemeSkeleton from './PhonemeSkeleton'

const SAMPLE_WORDS: Record<string, string[]> = {
  en: ['hello', 'world', 'please', 'thank you', 'good morning'],
  de: ['hallo', 'welt', 'bitte', 'danke', 'guten tag'],
  fr: ['bonjour', 'monde', 'merci', 's\'il vous plait', 'bonsoir'],
  es: ['hola', 'mundo', 'por favor', 'gracias', 'buenos dias'],
  it: ['ciao', 'mondo', 'per favore', 'grazie', 'buongiorno'],
  pt: ['ola', 'mundo', 'por favor', 'obrigado', 'bom dia'],
}

export default function DetectLanguageCard() {
  const [text, setText] = useState('hello world')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ text: string; language: string; supported_languages: string[] } | null>(null)
  const addToast = useToastStore(s => s.addToast)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const doDetect = useCallback(async (t: string) => {
    if (!t.trim()) { setResult(null); return }
    setLoading(true)
    try {
      const res = await phonemeController.detectLanguage(t)
      setResult(res)
    } catch {
      // silent — debounce will retry
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doDetect(text), 400)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [text, doDetect])

  const handleDetect = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    doDetect(text)
  }, [text, doDetect])

  useEffect(() => {
    const handleSubmit = () => handleDetect()
    window.addEventListener('phoneme-submit', handleSubmit)
    return () => window.removeEventListener('phoneme-submit', handleSubmit)
  }, [handleDetect])

  const handleSample = useCallback((lang: string, word: string) => {
    setText(word)
  }, [])

  const langLabel = (code: string) => PHONEME_LANGUAGES.find(l => l.value === code)?.label ?? code

  return (
    <Card>
      <CardHeader>
        <CardTitle>Detect Language</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <Input
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Enter text to detect language..."
            onKeyDown={e => e.key === 'Enter' && handleDetect()}
            className="flex-1"
          />
          <Button onClick={handleDetect} disabled={loading || !text.trim()}>
            {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
            Detect
          </Button>
        </div>

        {loading && !result && (
          <div className="space-y-3 p-4 rounded-lg bg-muted/30">
            <div className="flex items-center gap-3">
              <Skeleton className="h-4 w-16" />
              <Skeleton className="h-7 w-28 rounded-full" />
              <Skeleton className="h-4 w-8" />
            </div>
            <div className="space-y-1">
              <Skeleton className="h-4 w-36" />
              <div className="flex flex-wrap gap-1">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-6 w-16 rounded-full" />
                ))}
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="p-4 rounded-lg bg-muted/50 space-y-3 animate-in fade-in slide-in-from-top-1 duration-200">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-muted-foreground">Detected:</span>
              <Badge variant="default" className="text-base">{langLabel(result.language)}</Badge>
              <span className="text-sm text-muted-foreground">({result.language})</span>
            </div>

            <div>
              <p className="text-sm font-medium text-muted-foreground mb-1">Supported Languages:</p>
              <div className="flex flex-wrap gap-1">
                {result.supported_languages.map(lang => (
                  <Badge key={lang} variant="outline">{langLabel(lang)}</Badge>
                ))}
              </div>
            </div>
          </div>
        )}

        <div>
          <p className="text-sm font-medium text-muted-foreground mb-2">Try a sample:</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(SAMPLE_WORDS).flatMap(([lang, words]) =>
              words.slice(0, 2).map(word => (
                <Button
                  key={`${lang}-${word}`}
                  variant="secondary"
                  size="sm"
                  onClick={() => handleSample(lang, word)}
                >
                  <Badge variant="outline" className="mr-1">{lang.toUpperCase()}</Badge>
                  {word}
                </Button>
              ))
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
