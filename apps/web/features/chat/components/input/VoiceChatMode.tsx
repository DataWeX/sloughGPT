'use client'

import { useEffect, useState, useRef } from 'react'
import { useVoiceChat, VoiceExchange, VoiceSettings } from '@/features/chat/hooks/useVoiceChat'
import { VoiceWaveform, VoiceOrb, ListeningIndicator, ListeningBars } from '@/features/chat/components/input/VoiceWaveform'
import { cn } from '@sloughgpt/strui'
import { IconX, IconRefresh, IconSettings, IconSpeaker, IconMicFilled } from '@sloughgpt/strui'
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
    settings,
    availableVoices,
    updateSettings,
    handleToggle,
    startListening,
  } = useVoiceChat({ onMessage })

  const scrollRef = useRef<HTMLDivElement>(null)
  const [showTranscript, setShowTranscript] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => { startListening() }, [startListening])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [conversation, responseText])

  const isListening = state === 'listening'
  const isProcessing = state === 'processing'
  const isSpeaking = state === 'speaking'
  const hasConversation = conversation.length > 0 || responseText.length > 0

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <div className={cn('w-2 h-2 rounded-full', isListening ? 'bg-primary animate-pulse' :
            isSpeaking ? 'bg-emerald-500 animate-pulse' :
            isProcessing ? 'bg-amber-500 animate-pulse' :
            'bg-muted-foreground/30')} />
          <span className="text-sm font-medium" aria-live="polite">
            {isListening ? 'Listening' :
             isProcessing ? 'Processing...' :
             isSpeaking ? 'Speaking' : 'Ready'}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setShowTranscript(!showTranscript)}
            className={cn('p-2 rounded-lg transition-colors', showTranscript ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground')}
            aria-label={showTranscript ? 'Hide transcript' : 'Show transcript'}
          >
            {showTranscript ? 'Hide transcript' : 'Show transcript'}
          </button>
          <button
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            className={cn('p-2 rounded-lg transition-colors', showSettings ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground')}
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
        <ListeningIndicator micLevel={micLevel} active={isListening} />

        {showTranscript && hasConversation && (
          <VoiceTranscript
            ref={scrollRef}
            conversation={conversation}
            responseText={responseText}
            isSpeaking={isSpeaking}
          />
        )}

        <div className="mb-4">
          <VoiceWaveform
            level={micLevel}
            bars={32}
            variant={isListening ? 'mic' : isSpeaking ? 'speaker' : 'idle'}
            width={280}
            height={80}
          />
        </div>

        <div className="mb-4">
          <ListeningBars micLevel={micLevel} active={isListening} />
        </div>

        <VoiceOrb state={state} micLevel={micLevel}>
          <button
            type="button"
            onClick={handleToggle}
            disabled={isProcessing}
            className={cn('w-full h-full rounded-full flex items-center justify-center transition-all duration-200', isListening ? 'bg-primary text-primary-foreground' : isProcessing ? 'bg-muted text-muted-foreground cursor-wait' : 'bg-primary/10 text-primary hover:bg-primary/20')}
            aria-label={isListening ? 'Tap to stop listening' : 'Tap to start listening'}
          >
            {isProcessing ? (
              <IconRefresh className="h-8 w-8 animate-spin" />
            ) : isSpeaking ? (
              <IconSpeaker className="h-10 w-10" />
            ) : (
              <IconMicFilled className="h-10 w-10" />
            )}
          </button>
        </VoiceOrb>

        {/* Interim text */}
        {interimText && (
          <div className="mt-4 text-sm text-muted-foreground italic max-w-md text-center" aria-live="polite">
            {interimText}
          </div>
        )}

        {/* Final text */}
        {finalText && (
          <div className="mt-2 text-sm text-foreground max-w-md text-center">
            {finalText}
          </div>
        )}

        {/* Error message */}
        {errorMessage && (
          <div className="mt-4 px-4 py-2 rounded-lg bg-destructive/10 text-destructive text-sm max-w-md text-center">
            {errorMessage}
          </div>
        )}

        {/* Hint text */}
        {!interimText && !finalText && !errorMessage && (
          <div className="mt-4 text-sm text-muted-foreground italic max-w-md text-center" aria-live="polite">
            Speak naturally...
          </div>
        )}
      </div>

      {/* Footer hint */}
      <div className="px-4 py-3 text-center text-xs text-muted-foreground border-t border-border/50">
        {hasConversation ? `${conversation.length} exchange${conversation.length !== 1 ? 's' : ''}` : 'No conversation yet'}
      </div>
    </div>
  )
}
