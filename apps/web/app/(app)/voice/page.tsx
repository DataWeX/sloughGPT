'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Textarea, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'
import { VoicePresetCard } from '@/components/voice/VoicePresetCard'
import { useToastStore } from '@/lib/toast-store'

export default function VoicePage() {
  const [status, setStatus] = useState<VoiceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [ttsText, setTtsText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [lastResult, setLastResult] = useState<{ duration_ms: number; backend: string; sample_rate: number } | null>(null)
  const [ttsError, setTtsError] = useState<string | null>(null)
  const [ttsCount, setTtsCount] = useState(0)
  const [activePreset, setActivePreset] = useState<{ rate: number; pitch: number; voice: string } | null>(null)
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
      setTtsCount(c => c + 1)
      if (data.audio && data.backend === 'hf-model') {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`)
        audioRef.current = audio
        audio.play().catch(() => {}) // autoplay policy — expected
      } else if (data.backend === 'browser-fallback') {
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel()
          const utterance = new SpeechSynthesisUtterance(ttsText)
          if (activePreset) {
            utterance.rate = activePreset.rate
            utterance.pitch = activePreset.pitch
            if (activePreset.voice) {
              const match = window.speechSynthesis.getVoices().find(v => v.name === activePreset.voice)
              if (match) utterance.voice = match
            }
          }
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
      <PageContainer title="Voice" subtitle="Text-to-speech settings" loadingCards={3}>
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
        </KpiGrid>
        <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        <Card><CardContent><div className="h-40 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Voice" subtitle="Text-to-speech via browser speech synthesis">
      <KpiGrid>
        <StatCard
          label="Server TTS"
          value={status?.server_tts ? 'Available' : 'Not supported'}
        />
        <StatCard
          label="Engine"
          value={status?.model ? `Server (${status.model})` : 'Browser SpeechSynthesis'}
        />
        <StatCard label="TTS Calls" value={ttsCount} />
      </KpiGrid>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Text-to-Speech</CardTitle>
          <Button size="sm" variant="ghost" onClick={handleRefreshStatus}>
            <IconRefresh className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground bg-muted/30 rounded-md p-2">
              Server-side TTS requires the transformers library (not available).
              Text is spoken using your browser&apos;s built-in speech synthesis.
            </div>
          </div>
        </CardContent>
      </Card>

      <VoicePresetCard onApply={(p) => setActivePreset(p)} />

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
    </PageContainer>
  )
}
