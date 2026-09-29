'use client'

import { useState, useEffect } from 'react'
import { cn } from '@sloughgpt/strui'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'

export function DevVoiceTestCard() {
  const [status, setStatus] = useState<VoiceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [ttsText, setTtsText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [lastResult, setLastResult] = useState<{ duration_ms: number; backend: string } | null>(
    null,
  )
  const [ttsError, setTtsError] = useState<string | null>(null)

  useEffect(() => {
    voiceController
      .getStatus()
      .then((d) => setStatus(d))
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
      <div className="rounded-xl border border-white/[0.06] bg-background overflow-hidden">
        <div className="flex items-center h-9 px-4 bg-muted border-b border-white/[0.06]">
          <span className="text-[11px] font-medium text-muted-foreground">Voice Status</span>
        </div>
        <div className="px-4 py-3">
          {loading ? (
            <div className="h-14 w-full rounded-lg bg-muted animate-pulse" />
          ) : status ? (
            <div className="space-y-2.5">
              <div className="flex items-center gap-2.5">
                <span
                  className={cn(
                    'w-2 h-2 rounded-full',
                    status.server_tts ? 'bg-success' : 'bg-destructive',
                  )}
                />
                <span className="text-[12px] text-foreground">
                  {status.server_tts ? 'TTS Available' : 'TTS Unavailable'}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground font-mono">
                {status.model ?? 'no model'}
              </p>
              {status.error && (
                <div className="flex items-start gap-2 text-destructive bg-destructive/[0.08] rounded-lg px-3 py-2 text-[11px]">
                  <span className="shrink-0 text-[10px] font-bold">!</span>
                  {status.error}
                </div>
              )}
            </div>
          ) : (
            <p className="text-[11px] text-muted-foreground">Could not load status</p>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-white/[0.06] bg-background overflow-hidden">
        <div className="flex items-center h-9 px-4 bg-muted border-b border-white/[0.06]">
          <span className="text-[11px] font-medium text-muted-foreground">Quick Test</span>
        </div>
        <div className="px-4 py-3 space-y-2.5">
          <textarea
            placeholder="Type text to speak..."
            value={ttsText}
            onChange={(e) => setTtsText(e.target.value)}
            className="w-full h-16 rounded-lg border border-white/[0.06] bg-background px-3 py-2 text-[12px] text-foreground placeholder:text-muted-foreground resize-none outline-none focus:border-white/[0.12] transition-colors font-mono"
            aria-label="Text to speech input"
          />
          <button
            type="button"
            onClick={handleGenerate}
            disabled={generating || !ttsText.trim()}
            className={cn(
              'w-full h-8 rounded-lg text-[11px] font-medium transition-all duration-200',
              generating || !ttsText.trim()
                ? 'bg-success/20 text-success/40 cursor-not-allowed'
                : 'bg-success/10 text-success hover:bg-success/20',
            )}
          >
            {generating ? 'Generating...' : 'Speak'}
          </button>
          {ttsError && (
            <div className="flex items-start gap-2 text-destructive bg-destructive/[0.08] rounded-lg px-3 py-2 text-[11px]">
              <span className="shrink-0 text-[10px] font-bold">!</span>
              {ttsError}
            </div>
          )}
          {lastResult && (
            <p className="text-[10px] text-muted-foreground font-mono">
              {lastResult.duration_ms}ms · {lastResult.backend}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
