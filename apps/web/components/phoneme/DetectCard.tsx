'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'

export default function DetectCard() {
  const [text, setText] = useState('hello world')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ language: string; supported_languages: string[] } | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleDetect = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await phonemeController.detectLanguage(text)
      setResult(res)
    } catch {
      addToast('Detection failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [text, addToast])

  const langLabel = (code: string) =>
    PHONEME_LANGUAGES.find(l => l.value === code)?.label ?? code.toUpperCase()

  return (
    <Card>
      <CardHeader>
        <CardTitle>Detect Language</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-3">
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

        {result && (
          <div className="p-4 rounded-lg bg-muted/50 space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-muted-foreground">Detected:</span>
              <Badge variant="default" className="text-lg px-3 py-1">
                {langLabel(result.language)}
              </Badge>
              <code className="text-xs text-muted-foreground">({result.language})</code>
            </div>

            <div>
              <p className="text-sm font-medium text-muted-foreground mb-1">Supported Languages:</p>
              <div className="flex flex-wrap gap-1">
                {result.supported_languages.map(lang => (
                  <Badge
                    key={lang}
                    variant={lang === result.language ? 'default' : 'outline'}
                  >
                    {langLabel(lang)}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
