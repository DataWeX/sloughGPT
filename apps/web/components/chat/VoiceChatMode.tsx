'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { chatController } from '@/lib/chat-controller'
import { IconX, IconRefresh } from '@/components/ui'

// ── Speech Recognition Types (local, no global conflicts) ─────────────────

interface SRResult {
  isFinal: boolean
  [index: number]: { transcript: string; confidence: number }
}

interface SRResultList {
  length: number
  [index: number]: SRResult
}

interface SREvent extends Event {
  results: SRResultList
  resultIndex: number
}

interface SRInstance {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onstart: (() => void) | null
  onend: (() => void) | null
  onerror: ((e: Event) => void) | null
  onresult: ((e: SREvent) => void) | null
}

interface SRConstructor {
  new (): SRInstance
}

// ── Voice Chat Mode Component ─────────────────────────────────────────────

interface VoiceChatModeProps {
  onMessage: (text: string) => void
  onClose: () => void
}

const SILENCE_TIMEOUT = 2000

export function VoiceChatMode({ onMessage, onClose }: VoiceChatModeProps) {
  const [isListening, setIsListening] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [interimText, setInterimText] = useState('')
  const [finalText, setFinalText] = useState('')
  const [responseText, setResponseText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [pulsePhase, setPulsePhase] = useState(0)

  const recognitionRef = useRef<SRInstance | null>(null)
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pulseRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isListeningRef = useRef(false)

  // Pulse animation
  useEffect(() => {
    if (isListening) {
      pulseRef.current = setInterval(() => setPulsePhase(p => (p + 1) % 360), 50)
    } else {
      if (pulseRef.current) clearInterval(pulseRef.current)
      setPulsePhase(0)
    }
    return () => { if (pulseRef.current) clearInterval(pulseRef.current) }
  }, [isListening])

  const resetSilenceTimer = useCallback((text: string) => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    silenceTimerRef.current = setTimeout(() => {
      if (isListeningRef.current && text.trim()) {
        stopListening()
        handleSubmit(text.trim())
      }
    }, SILENCE_TIMEOUT)
  }, [])

  const startListening = useCallback(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) {
      setError('Speech recognition not supported — try Chrome or Safari')
      return
    }

    setFinalText('')
    setInterimText('')
    setResponseText('')
    setError(null)

    const recognition: SRInstance = new SR()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => {
      setIsListening(true)
      isListeningRef.current = true
    }

    recognition.onend = () => {
      setIsListening(false)
      isListeningRef.current = false
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    }

    recognition.onerror = (e: Event) => {
      const type = (e as any)?.error || 'unknown'
      if (type === 'not-allowed') {
        setError('Microphone access denied')
      } else if (type !== 'aborted') {
        setError(`Speech error: ${type}`)
      }
      setIsListening(false)
      isListeningRef.current = false
    }

    recognition.onresult = (event: SREvent) => {
      let interim = ''
      let final = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          final += result[0].transcript
        } else {
          interim += result[0].transcript
        }
      }
      if (final) {
        const newFinal = finalText + final
        setFinalText(newFinal)
        setInterimText('')
        resetSilenceTimer(newFinal)
      }
      if (interim) {
        setInterimText(interim)
        resetSilenceTimer(finalText + interim)
      }
    }

    recognitionRef.current = recognition
    recognition.start()
  }, [finalText, resetSilenceTimer])

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
    }
    isListeningRef.current = false
    setIsListening(false)
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
  }, [])

  const handleSubmit = useCallback(async (text: string) => {
    if (!text || isProcessing) return
    setIsProcessing(true)
    setResponseText('')
    setError(null)

    onMessage(text)

    try {
      let fullResponse = ''
      for await (const token of chatController.stream(text)) {
        fullResponse += token
        setResponseText(fullResponse)
      }
    } catch (e: any) {
      setError(e.message || 'Generation failed')
    } finally {
      setIsProcessing(false)
      // Auto-resume listening after response
      setTimeout(() => startListening(), 500)
    }
  }, [isProcessing, onMessage, startListening])

  const handleToggle = useCallback(() => {
    if (isListening) {
      stopListening()
    } else if (!isProcessing) {
      startListening()
    }
  }, [isListening, isProcessing, startListening, stopListening])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening()
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    }
  }, [stopListening])

  // Auto-start on mount
  useEffect(() => {
    startListening()
  }, [startListening])

  const hasInterim = interimText.length > 0

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col items-center justify-center">
      {/* Close button */}
      <button
        onClick={onClose}
        className="absolute top-4 right-4 p-2 rounded-full hover:bg-muted transition-colors"
        aria-label="Exit voice mode"
      >
        <IconX className="h-5 w-5" />
      </button>

      {/* Main content */}
      <div className="flex flex-col items-center gap-8 max-w-lg w-full px-6">
        {/* Listening orb */}
        <div className="relative">
          {/* Pulse rings */}
          {isListening && (
            <>
              <div
                className="absolute inset-0 rounded-full border-2 border-primary/30"
                style={{
                  transform: `scale(${1 + Math.sin(pulsePhase * Math.PI / 180) * 0.3})`,
                  opacity: 0.5 - Math.sin(pulsePhase * Math.PI / 180) * 0.3,
                }}
              />
              <div
                className="absolute inset-0 rounded-full border border-primary/20"
                style={{
                  transform: `scale(${1 + Math.sin((pulsePhase + 120) * Math.PI / 180) * 0.5})`,
                  opacity: 0.3,
                }}
              />
            </>
          )}

          {/* Main button */}
          <button
            onClick={handleToggle}
            disabled={isProcessing}
            className={`
              relative z-10 w-24 h-24 rounded-full flex items-center justify-center
              transition-all duration-300 shadow-lg
              ${isListening
                ? 'bg-primary text-primary-foreground scale-110'
                : isProcessing
                  ? 'bg-muted text-muted-foreground cursor-wait'
                  : 'bg-primary/10 text-primary hover:bg-primary/20'
              }
            `}
            aria-label={isListening ? 'Tap to stop listening' : 'Tap to start listening'}
          >
            {isProcessing ? (
              <IconRefresh className="h-8 w-8 animate-spin" />
            ) : (
              <svg className="h-10 w-10" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
              </svg>
            )}
          </button>
        </div>

        {/* Status text */}
        <div className="text-center space-y-2">
          <p className="text-lg font-medium">
            {isListening
              ? 'Listening...'
              : isProcessing
                ? 'Thinking...'
                : responseText
                  ? 'Tap to continue'
                  : 'Tap microphone to start'
            }
          </p>
          {hasInterim && (
            <p className="text-sm text-muted-foreground">
              {finalText}
              <span className="text-primary/70">{interimText}</span>
            </p>
          )}
          {!hasInterim && finalText && (
            <p className="text-sm text-muted-foreground">{finalText}</p>
          )}
        </div>

        {/* Response display */}
        {responseText && (
          <div className="w-full bg-muted/50 rounded-xl p-4 text-sm leading-relaxed max-h-48 overflow-y-auto">
            {responseText}
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="w-full bg-error/10 text-error rounded-xl p-4 text-sm text-center">
            {error}
          </div>
        )}

        {/* Instructions */}
        <p className="text-xs text-muted-foreground/60 text-center">
          Speak naturally — auto-sends after 2s of silence.
          <br />
          Tap orb to toggle listening.
        </p>
      </div>
    </div>
  )
}
