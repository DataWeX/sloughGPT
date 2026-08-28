'use client'

import { useState, useCallback, useEffect, useRef, memo } from 'react'
import { Button, IconMicFilled, IconPlay, IconStop } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import { estimateTokens } from '@/lib/format-bytes'

interface VoiceInputAgentProps {
  onTranscript: (text: string) => void
  onSend?: () => void
  model?: string
  className?: string
}

interface AudioStats {
  duration: number
  words: number
  tokens: number
  estimatedCost: number
}

const MODEL_PRICING: Record<string, { input: number; output: number }> = {
  'gpt-4': { input: 0.03, output: 0.06 },
  'gpt-4-turbo': { input: 0.01, output: 0.03 },
  'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
  'whisper-1': { input: 0.006, output: 0 },
}

export const VoiceInputAgent = memo(function VoiceInputAgent({
  onTranscript,
  onSend,
  model = 'whisper-1',
  className,
}: VoiceInputAgentProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [duration, setDuration] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [audioStats, setAudioStats] = useState<AudioStats | null>(null)
  const [showStats, setShowStats] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const startTimeRef = useRef<number>(0)

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const audioUrl = URL.createObjectURL(blob)

        // Simulate transcription stats
        const durationSec = duration
        const words = Math.floor(durationSec * 2.5) // ~150 words per minute
        const tokens = estimateTokens(Array(words).fill('word').join(' '))
        const pricing = MODEL_PRICING[model] || MODEL_PRICING['whisper-1']
        const estimatedCost = (durationSec / 60) * pricing.input

        setAudioStats({
          duration: durationSec,
          words,
          tokens,
          estimatedCost,
        })

        stream.getTracks().forEach(t => t.stop())
        URL.revokeObjectURL(audioUrl)
      }

      mediaRecorder.start()
      setIsRecording(true)
      setIsPaused(false)
      setDuration(0)
      setTranscript('')
      setError(null)
      startTimeRef.current = Date.now()

      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 1000)
    } catch (err) {
      setError('Microphone access denied')
    }
  }, [model, duration])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    setIsRecording(false)
    setIsPaused(false)
  }, [])

  const pauseRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.pause()
      setIsPaused(true)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const resumeRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'paused') {
      mediaRecorderRef.current.resume()
      setIsPaused(false)
      startTimeRef.current = Date.now() - duration * 1000
      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 1000)
    }
  }, [duration])

  const handleSendTranscript = useCallback(() => {
    if (transcript.trim()) {
      onTranscript(transcript.trim())
      if (onSend) onSend()
      setTranscript('')
      setAudioStats(null)
    }
  }, [transcript, onTranscript, onSend])

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const formatCost = (cost: number) => {
    if (cost < 0.001) return '<$0.001'
    return `$${cost.toFixed(4)}`
  }

  return (
    <div className={cn('border rounded-xl bg-card overflow-hidden', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <div className={cn(
            'w-2 h-2 rounded-full',
            isRecording ? 'bg-destructive animate-pulse' : 'bg-muted',
          )} />
          <span className="text-xs font-medium">
            {isRecording ? (isPaused ? 'Paused' : 'Recording') : 'Voice Input'}
          </span>
        </div>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
          {model}
        </span>
      </div>

      {/* Main area */}
      <div className="p-4">
        {/* Timer */}
        <div className="text-center mb-4">
          <span className={cn(
            'text-3xl font-mono',
            isRecording ? 'text-destructive' : 'text-muted-foreground',
          )}>
            {formatDuration(duration)}
          </span>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-center gap-3 mb-4">
          {!isRecording ? (
            <Button
              variant="default"
              size="lg"
              className="rounded-full h-12 w-12 p-0"
              onClick={startRecording}
              aria-label="Start recording"
            >
              <IconMicFilled className="h-6 w-6" aria-hidden="true" />
            </Button>
          ) : (
            <>
              {!isPaused ? (
                <Button
                  variant="outline"
                  size="lg"
                  className="rounded-full h-10 w-10 p-0"
                  onClick={pauseRecording}
                >
                  <IconStop className="h-4 w-4" aria-hidden="true" />
                </Button>
              ) : (
                <Button
                  variant="outline"
                  size="lg"
                  className="rounded-full h-10 w-10 p-0"
                  onClick={resumeRecording}
                >
                  <IconPlay className="h-4 w-4" aria-hidden="true" />
                </Button>
              )}
              <Button
                variant="destructive"
                size="lg"
                className="rounded-full h-12 w-12 p-0"
                onClick={stopRecording}
              >
                <IconStop className="h-6 w-6" aria-hidden="true" />
              </Button>
            </>
          )}
        </div>

        {/* Transcript area */}
        {transcript && (
          <div className="mb-4">
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              className="w-full h-24 text-sm bg-transparent border rounded-lg px-3 py-2 resize-none focus:outline-none focus:ring-1 focus:ring-primary/50"
              placeholder="Transcript will appear here..."
            />
            <div className="flex justify-end mt-2">
              <Button
                size="sm"
                onClick={handleSendTranscript}
                disabled={!transcript.trim()}
              >
                Send Transcript
              </Button>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="text-center text-destructive text-xs mb-4">
            {error}
          </div>
        )}
      </div>

      {/* Stats bar */}
      {showStats && audioStats && (
        <div className="flex items-center justify-between px-3 py-2 border-t bg-muted/20">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">Duration:</span>
              <span className="text-[10px] font-medium">{formatDuration(audioStats.duration)}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">Words:</span>
              <span className="text-[10px] font-medium">{audioStats.words}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">Tokens:</span>
              <span className="text-[10px] font-medium">{audioStats.tokens}</span>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-muted-foreground">Est. cost:</span>
            <span className="text-[10px] font-medium text-primary">
              {formatCost(audioStats.estimatedCost)}
            </span>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between px-3 py-2 border-t">
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-6"
          onClick={() => setShowStats(!showStats)}
        >
          {showStats ? 'Hide Stats' : 'Stats'}
        </Button>
        <div className="text-[10px] text-muted-foreground/40">
          {isRecording ? 'Click stop to end recording' : 'Click mic to start recording'}
        </div>
      </div>
    </div>
  )
})