'use client'

import { useEffect, useState } from 'react'
import { useVoiceChat } from '@/hooks/useVoiceChat'
import { IconX, IconRefresh } from '@/components/ui'

interface VoiceChatModeProps {
  onMessage: (text: string) => void
  onClose: () => void
}

export function VoiceChatMode({ onMessage, onClose }: VoiceChatModeProps) {
  const {
    state,
    interimText,
    finalText,
    responseText,
    errorMessage,
    handleToggle,
    startListening,
  } = useVoiceChat({ onMessage })

  const [pulsePhase, setPulsePhase] = useState(0)
  const isListening = state === 'listening'
  const isProcessing = state === 'processing'
  const isSpeaking = state === 'speaking'

  // Auto-start on mount
  useEffect(() => { startListening() }, [startListening])

  // Pulse animation
  useEffect(() => {
    if (isListening || isSpeaking) {
      const interval = setInterval(() => setPulsePhase(p => (p + 1) % 360), 50)
      return () => clearInterval(interval)
    }
    setPulsePhase(0)
  }, [isListening, isSpeaking])

  const hasInterim = interimText.length > 0

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col items-center justify-center">
      <button
        onClick={onClose}
        className="absolute top-4 right-4 p-2 rounded-full hover:bg-muted transition-colors"
        aria-label="Exit voice mode"
      >
        <IconX className="h-5 w-5" />
      </button>

      <div className="flex flex-col items-center gap-8 max-w-lg w-full px-6">
        {/* Orb */}
        <div className="relative">
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

          {isSpeaking && (
            <>
              <div
                className="absolute inset-0 rounded-full border-2 border-success/30"
                style={{
                  transform: `scale(${1 + Math.sin(pulsePhase * Math.PI / 180) * 0.2})`,
                  opacity: 0.6,
                }}
              />
              <div
                className="absolute inset-0 rounded-full border border-success/20"
                style={{
                  transform: `scale(${1 + Math.sin((pulsePhase + 180) * Math.PI / 180) * 0.35})`,
                  opacity: 0.3,
                }}
              />
            </>
          )}

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
            ) : isSpeaking ? (
              <svg className="h-10 w-10" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
              </svg>
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
                : isSpeaking
                  ? 'Speaking...'
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

        {/* Response */}
        {responseText && (
          <div className="w-full bg-muted/50 rounded-xl p-4 text-sm leading-relaxed max-h-48 overflow-y-auto">
            {responseText}
          </div>
        )}

        {/* Error */}
        {errorMessage && (
          <div className="w-full bg-error/10 text-error rounded-xl p-4 text-sm text-center">
            {errorMessage}
          </div>
        )}

        <p className="text-xs text-muted-foreground/60 text-center">
          Speak naturally — auto-sends after 2s of silence.
          <br />
          Tap orb to toggle listening.
        </p>
      </div>
    </div>
  )
}
