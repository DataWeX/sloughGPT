'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Textarea, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'
import { VoicePresetCard } from '@/components/voice/VoicePresetCard'
import { useToastStore } from '@/lib/toast-store'

export function VoiceSection() {
  const [status, setStatus] = useState<VoiceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [ttsText, setTtsText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [lastResult, setLastResult] = useState<{ duration_ms: number; backend: string; sample_rate: number } | null>(null)
  const [ttsError, setTtsError] = useState<string | null>(null)
  const [ttsCount, setTtsCount] = useState(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    voiceController.getStatus()
      .then(d => setStatus(d))
      .catch(() => { addToast('Could not load voice status', 'error') })
      .finally(() => setLoading(false))
  }, [])

  const handleRefreshStatus = async () => {
    try {
      setStatus(await voiceController.getStatus())
    } catch {
      addToast('Could not refresh voice status', 'error')
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
      setTtsCount(c => c + 1)
      if (data.audio && data.backend === 'hf-model') {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`)
        audioRef.current = audio
        audio.play().catch(() => {}) // autoplay policy — expected
      } else if (data.backend === 'browser-fallback') {
        if ('speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(ttsText)
          window.speechSynthesis.speak(utterance)
        }
      }
    } catch (err) {
      setTtsError(err instanceof Error ? err.message : 'Could not tts')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <>
      <div className="flex items-center justify-between border-b border-border/30 pb-2 pt-1">
        <h2 className="text-base font-medium">Text to Speech</h2>
      </div>

      {loading ? (
        <KpiGrid>
          <StatCard label="Loading" value="..." />
          <StatCard label="Loading" value="..." />
          <StatCard label="Loading" value="..." />
        </KpiGrid>
      ) : (
        <KpiGrid>
          <StatCard label="Text-to-Speech" value={status?.server_tts ? 'Available' : 'Unavailable'} />
          <StatCard label="Model" value={status?.model ?? 'None'} />
          <StatCard label="TTS Calls" value={ttsCount} />
        </KpiGrid>
      )}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">TTS Backend</CardTitle>
          <Button size="sm" variant="ghost" onClick={handleRefreshStatus}>
            <IconRefresh className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent>
          {status ? (
            <div className="space-y-3">
              <KpiGrid columns={4}>
                <StatCard label="Service" value={<span className={status.server_tts ? 'text-success' : 'text-muted-foreground'}>{status.server_tts ? 'Online' : 'Offline'}</span>} />
                <StatCard label="Model" value={status.model ?? '—'} />
                <StatCard label="Fallback" value="Browser" />
                <StatCard label="Status" value={<span className={status.error ? 'text-destructive' : 'text-success'}>{status.error ? 'Error' : 'Ready'}</span>} />
              </KpiGrid>
              {status.error && (
                <div className="text-xs text-destructive bg-destructive/5 rounded-md p-2">{status.error}</div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Could not load TTS status.</p>
          )}
        </CardContent>
      </Card>

      <VoicePresetCard />

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
            <p>Text-to-speech uses HuggingFace bark-small model when available.</p>
            <p>Falls back to browser native speechSynthesis if the model is unavailable.</p>
            <p>Voice input is available in the chat page via the microphone button.</p>
          </div>
        </CardContent>
      </Card>
    </>
  )
}
