'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Textarea } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'
import { useToastStore } from '@/lib/toast-store'

export default function VoicePage() {
  const [status, setStatus] = useState<VoiceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [ttsText, setTtsText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [lastResult, setLastResult] = useState<{ duration_ms: number; backend: string; sample_rate: number } | null>(null)
  const [ttsError, setTtsError] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    voiceController.getStatus()
      .then(d => setStatus(d))
      .catch(() => { addToast('Failed to load voice status', 'error') })
      .finally(() => setLoading(false))
  }, [])

  const handleRefreshStatus = async () => {
    try {
      setStatus(await voiceController.getStatus())
    } catch {
      addToast('Failed to refresh voice status', 'error')
    }
  }

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
      setLastResult({ duration_ms: data.duration_ms, backend: data.backend, sample_rate: data.sample_rate })
      if (data.audio && data.backend === 'hf-model') {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`)
        audioRef.current = audio
        audio.play().catch(() => {})
      } else if (data.backend === 'browser-fallback') {
        if ('speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(ttsText)
          window.speechSynthesis.speak(utterance)
        }
      }
    } catch (err) {
      setTtsError(err instanceof Error ? err.message : 'TTS failed')
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Voice" subtitle="Text-to-speech settings" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Voice" subtitle="Text-to-speech settings" />} />
      <div className="space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">TTS Backend</CardTitle>
            <Button size="sm" variant="ghost" onClick={handleRefreshStatus}>
              <IconRefresh className="h-3.5 w-3.5" />
            </Button>
          </CardHeader>
          <CardContent>
            {status ? (
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">Server TTS</div>
                  <div className={`text-sm font-mono font-medium ${status.server_tts ? 'text-success' : 'text-muted-foreground'}`}>
                    {status.server_tts ? 'Available' : 'Unavailable'}
                  </div>
                </div>
                <div className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">Model</div>
                  <div className="text-sm font-mono font-medium">{status.model ?? '—'}</div>
                </div>
                <div className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">Fallback</div>
                  <div className="text-sm font-mono font-medium text-muted-foreground">Browser TTS</div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Could not load TTS status.</p>
            )}
            {status?.error && (
              <div className="mt-2 text-xs text-destructive">{status.error}</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Test TTS</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea
              value={ttsText}
              onChange={e => setTtsText(e.target.value)}
              placeholder="Enter text to speak..."
              rows={3}
            />
            <div className="flex items-center gap-3">
              <Button size="sm" onClick={handleGenerate} disabled={generating || !ttsText.trim()}>
                {generating ? 'Generating...' : 'Generate & Play'}
              </Button>
              {lastResult && (
                <span className="text-xs text-muted-foreground">
                  {lastResult.backend} · {lastResult.duration_ms}ms · {lastResult.sample_rate}Hz
                </span>
              )}
            </div>
            {ttsError && (
              <div className="text-xs text-destructive">{ttsError}</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">About</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground space-y-1">
              <p>Server-side TTS uses HuggingFace bark-small model when available.</p>
              <p>Falls back to browser native speechSynthesis if server model is unavailable.</p>
              <p>Voice input is available in the chat page via the microphone button.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
