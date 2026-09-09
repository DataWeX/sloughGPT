'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { phonemeController, type PhonemeSynthesizeResult } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'
import WaveformCanvas from './WaveformCanvas'
import SpectrogramCanvas from './SpectrogramCanvas'

export default function SynthesisCard() {
  const [text, setText] = useState('hello world')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PhonemeSynthesizeResult | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleSynthesize = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await phonemeController.synthesize(text)
      setResult(res)
    } catch (err) {
      addToast('Synthesis failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [text, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Synthesize Speech</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-3">
          <Input
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Enter text to synthesize..."
            onKeyDown={e => e.key === 'Enter' && handleSynthesize()}
            className="flex-1"
          />
          <Button onClick={handleSynthesize} disabled={loading || !text.trim()}>
            {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
            Synthesize
          </Button>
        </div>

        {result && (
          <div className="space-y-3 p-4 rounded-lg bg-muted/50">
            <div className="text-sm text-muted-foreground">
              Duration: {result.duration_sec.toFixed(2)}s | Generated in {result.elapsed_ms.toFixed(0)}ms
            </div>
            <audio controls className="w-full" src={result.audio} />
            <WaveformCanvas audioSrc={result.audio} />
            {result.spectrogram && (
              <SpectrogramCanvas spectrogram={result.spectrogram} />
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
