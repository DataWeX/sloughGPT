'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Textarea, cn } from '@sloughgpt/strui'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'

export function DevVoiceTestCard() {
  const [status, setStatus] = useState<VoiceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [ttsText, setTtsText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [lastResult, setLastResult] = useState<{ duration_ms: number; backend: string } | null>(null)
  const [ttsError, setTtsError] = useState<string | null>(null)

  useEffect(() => {
    voiceController.getStatus()
      .then(d => setStatus(d))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleGenerate = async () => {
    if (!ttsText.trim()) return
    setGenerating(true)
    setTtsError(null)
    setLastResult(null)
    try {
      const data = await voiceController.tts(ttsText)
      if (data.detail) {
        setTtsError(data.detail)
        return
      }
      setLastResult({ duration_ms: data.duration_ms, backend: data.backend })
      if (data.audio) {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`)
        audio.play().catch(() => {})
      }
    } catch {
      setTtsError('TTS request failed')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Voice Status</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="h-16 w-full animate-pulse bg-muted rounded" />
          ) : status ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className={cn(
                  'w-2 h-2 rounded-full',
                  status.server_tts ? 'bg-green-500' : 'bg-red-500',
                )} />
                <span className="text-xs">{status.server_tts ? 'TTS Available' : 'TTS Unavailable'}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Model: {status.model ?? 'none'}
              </p>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Could not load status</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quick Test</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Textarea
            placeholder="Type text to speak..."
            value={ttsText}
            onChange={e => setTtsText(e.target.value)}
            className="h-16 text-xs resize-none"
            aria-label="Text to speech input"
          />
          <Button
            size="sm"
            onClick={handleGenerate}
            disabled={generating || !ttsText.trim()}
            className="w-full"
          >
            {generating ? 'Generating...' : 'Speak'}
          </Button>
          {ttsError && (
            <p className="text-xs text-destructive">{ttsError}</p>
          )}
          {lastResult && (
            <p className="text-xs text-muted-foreground">
              {lastResult.duration_ms}ms · {lastResult.backend}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
