'use client'

import { useEffect, useState, useRef } from 'react'
import { useVoiceChat, VoiceExchange, VoiceSettings } from '@/features/chat/hooks/useVoiceChat'
import { VoiceOrb, voiceStateTone } from '@/features/chat/components/input/VoiceWaveform'
import { cn, Spinner } from '@sloughgpt/strui'
import { IconX, IconSettings, IconSpeaker, IconMicFilled } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { VoiceSettingsPanel } from './VoiceSettingsPanel'
import { VoiceTranscript } from './VoiceTranscript'

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
    conversation,
    micLevel,
    micLevelRef,
    settings,
    availableVoices,
    updateSettings,
    handleToggle,
    startListening,
  } = useVoiceChat({ onMessage })

  const scrollRef = useRef<HTMLDivElement>(null)
  const [showTranscript, setShowTranscript] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    startListening()
  }, [startListening])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [conversation, responseText])

  const isListening = state === 'listening'
  const isProcessing = state === 'processing'
  const isSpeaking = state === 'speaking'
  const hasConversation = conversation.length > 0 || responseText.length > 0
  const tone = voiceStateTone(state)

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <div
            className={cn('w-2 h-2 rounded-full', tone.dot, tone.pulse && 'animate-pulse')}
          />
          <span className="text-sm font-medium" aria-live="polite">
            {isListening
              ? 'Listening'
              : isProcessing
                ? 'Processing...'
                : isSpeaking
                  ? 'Speaking'
                  : 'Ready'}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setShowTranscript(!showTranscript)}
            className={cn(
              'p-2 rounded-lg transition-colors',
              showTranscript
                ? 'bg-primary/10 text-primary'
                : 'text-muted-foreground hover:text-foreground',
            )}
            aria-label={showTranscript ? 'Hide transcript' : 'Show transcript'}
          >
            {showTranscript ? 'Hide transcript' : 'Show transcript'}
          </button>
          <button
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            className={cn(
              'p-2 rounded-lg transition-colors',
              showSettings
                ? 'bg-primary/10 text-primary'
                : 'text-muted-foreground hover:text-foreground',
            )}
            aria-label="Voice settings"
          >
            <IconSettings className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg text-muted-foreground hover:text-foreground"
            aria-label="Exit voice mode"
          >
            <IconX className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <VoiceSettingsPanel
          settings={settings}
          availableVoices={availableVoices}
          updateSettings={updateSettings}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col items-center justify-center px-4 overflow-hidden relative">
        {showTranscript && hasConversation && (
          <VoiceTranscript
            ref={scrollRef}
            conversation={conversation}
            responseText={responseText}
            isSpeaking={isSpeaking}
          />
        )}

        {/* Hero orb — one visual surface for the whole voice state machine.
            Level arrives per-frame through micLevelRef (written to --mic-level
            on the orb's own rAF loop), so this tree only re-renders on
            discrete state changes. */}
        <VoiceOrb state={state} micLevel={micLevel} levelRef={micLevelRef}>
          <button
            type="button"
            onClick={handleToggle}
            disabled={isProcessing}
            className={cn(
              'relative z-10 w-full h-full rounded-full flex items-center justify-center transition-all duration-200',
              'focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background',
              'active:scale-[0.98]',
              isListening
                ? 'bg-primary text-primary-foreground hover:opacity-90'
                : isProcessing
                  ? 'bg-muted text-muted-foreground cursor-wait'
                  : 'bg-primary/10 text-primary hover:bg-primary/20',
            )}
            aria-label={isListening ? 'Tap to stop listening' : 'Tap to start listening'}
          >
            {isProcessing ? (
              <Spinner className="h-10 w-10" />
            ) : isSpeaking ? (
              <IconSpeaker className="h-12 w-12" />
            ) : (
              <IconMicFilled className="h-12 w-12" />
            )}
          </button>
        </VoiceOrb>

        {/* Interim text */}
        {interimText && (
          <div
            className="mt-4 text-sm text-muted-foreground italic max-w-md text-center"
            aria-live="polite"
          >
            {interimText}
          </div>
        )}

        {/* Final text */}
        {finalText && (
          <div className="mt-2 text-sm text-foreground max-w-md text-center">{finalText}</div>
        )}

        {/* Error message */}
        {errorMessage && (
          <div className="mt-4 max-w-md w-full">
            <StatusBanner
              variant="error"
              message={errorMessage}
              dismissible={false}
              className="text-center"
            />
          </div>
        )}

        {/* Hint text */}
        {!interimText && !finalText && !errorMessage && (
          <div
            className="mt-4 text-sm text-muted-foreground italic max-w-md text-center"
            aria-live="polite"
          >
            Speak naturally...
          </div>
        )}
      </div>

      {/* Footer hint */}
      <div className="px-4 py-3 text-center text-xs text-muted-foreground border-t border-border/50">
        {hasConversation
          ? `${conversation.length} exchange${conversation.length !== 1 ? 's' : ''}`
          : 'No conversation yet'}
      </div>
    </div>
  )
}
