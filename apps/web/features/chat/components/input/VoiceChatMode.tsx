'use client'

import { useEffect, useState, useRef } from 'react'
import { useVoiceChat, VoiceExchange, VoiceSettings } from '@/features/chat/hooks/useVoiceChat'
import { VoiceWaveform, VoiceOrb, ListeningIndicator, ListeningBars } from '@/features/chat/components/input/VoiceWaveform'
import { Slider } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { IconX, IconRefresh, IconSettings, IconSpeaker, IconMicFilled } from '@sloughgpt/strui'

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

  // Auto-start on mount
  useEffect(() => { startListening() }, [startListening])

  // Auto-scroll conversation to bottom
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
          <div className={`w-2 h-2 rounded-full ${
            isListening ? 'bg-primary animate-pulse' :
            isSpeaking ? 'bg-emerald-500 animate-pulse' :
            isProcessing ? 'bg-amber-500 animate-pulse' :
            'bg-muted-foreground/30'
          }`} />
          <span className="text-sm font-medium">
            {isListening ? 'Listening' :
             isProcessing ? 'Thinking' :
             isSpeaking ? 'Speaking' :
             'Voice Chat'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTranscript(!showTranscript)}
            className="text-xs px-2 py-1 rounded-md hover:bg-muted transition-colors text-muted-foreground"
            aria-expanded={showTranscript}
          >
            {showTranscript ? 'Hide' : 'Show'} transcript
          </button>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-1.5 rounded-full hover:bg-muted transition-colors"
            aria-label="Voice settings"
            aria-expanded={showSettings}
          >
            <IconSettings className="h-4 w-4" />
          </button>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full hover:bg-muted transition-colors"
            aria-label="Exit voice mode"
          >
            <IconX className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="border-b border-border/50 px-4 py-3 bg-muted/30">
          <div className="max-w-lg mx-auto space-y-3">
            <div className="flex items-center justify-between">
              <label htmlFor="voice-speech-rate" className="text-xs font-medium text-muted-foreground">Speech Rate</label>
              <span className="text-xs text-muted-foreground/60">{settings.rate.toFixed(1)}x</span>
            </div>
            <Slider
              id="voice-speech-rate"
              value={[settings.rate]}
              onValueChange={([v]) => updateSettings({ rate: v })}
              min={0.5}
              max={2.0}
              step={0.1}
            />

            <div className="flex items-center justify-between">
              <label htmlFor="voice-interrupt-sensitivity" className="text-xs font-medium text-muted-foreground">Interrupt Sensitivity</label>
              <span className="text-xs text-muted-foreground/60">
                {settings.interruptThreshold < 0.1 ? 'High' : settings.interruptThreshold < 0.2 ? 'Medium' : 'Low'}
              </span>
            </div>
            <Slider
              id="voice-interrupt-sensitivity"
              value={[settings.interruptThreshold]}
              onValueChange={([v]) => updateSettings({ interruptThreshold: v })}
              min={0.05}
              max={0.5}
              step={0.05}
            />

            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">Auto-resume Listening</label>
              <button
                onClick={() => updateSettings({ autoResume: !settings.autoResume })}
                role="switch"
                aria-checked={settings.autoResume}
                aria-label="Auto-resume listening"
                className={`w-9 h-5 rounded-full transition-colors ${
                  settings.autoResume ? 'bg-primary' : 'bg-muted-foreground/30'
                }`}
              >
                <div className={`w-4 h-4 rounded-full bg-white shadow transition-transform ${
                  settings.autoResume ? 'translate-x-4.5' : 'translate-x-0.5'
                }`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">Push-to-talk (Space)</label>
              <button
                onClick={() => updateSettings({ pushToTalk: !settings.pushToTalk })}
                className={`w-9 h-5 rounded-full transition-colors ${
                  settings.pushToTalk ? 'bg-primary' : 'bg-muted-foreground/30'
                }`}
              >
                <div className={`w-4 h-4 rounded-full bg-white shadow transition-transform ${
                  settings.pushToTalk ? 'translate-x-4.5' : 'translate-x-0.5'
                }`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">Stream speech (speak as ready)</label>
              <button
                onClick={() => updateSettings({ streamingTTS: !settings.streamingTTS })}
                className={`w-9 h-5 rounded-full transition-colors ${
                  settings.streamingTTS ? 'bg-primary' : 'bg-muted-foreground/30'
                }`}
              >
                <div className={`w-4 h-4 rounded-full bg-white shadow transition-transform ${
                  settings.streamingTTS ? 'translate-x-4.5' : 'translate-x-0.5'
                }`} />
              </button>
            </div>

            {availableVoices.length > 0 && (
              <>
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-muted-foreground">Voice</label>
                </div>
                <Select
                  value={settings.voiceName || ''}
                  onValueChange={(v) => updateSettings({ voiceName: v || null })}
                >
                  <SelectTrigger className="w-full text-xs">
                    <SelectValue placeholder="Default" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">Default</SelectItem>
                    {availableVoices.filter(v => v.lang.startsWith('en')).map(v => (
                      <SelectItem key={v.name} value={v.name}>{v.name} ({v.lang})</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </>
            )}
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col items-center justify-center px-4 overflow-hidden relative">
        {/* Listening animation — pulsing rings + center glow */}
        <ListeningIndicator micLevel={micLevel} active={isListening} />

        {/* Conversation transcript (collapsible) */}
        {showTranscript && hasConversation && (
          <div
            ref={scrollRef}
            className="w-full max-w-lg mb-6 max-h-64 overflow-y-auto rounded-xl border border-border/50 bg-muted/30 p-3 space-y-3"
          >
            {conversation.map((ex) => (
              <TranscriptExchange key={ex.id} exchange={ex} />
            ))}
            {responseText && state !== 'speaking' && (
              <div className="text-sm text-emerald-600 dark:text-emerald-400">
                <span className="font-medium text-xs text-muted-foreground">Assistant</span>
                <p className="mt-0.5">{responseText}</p>
              </div>
            )}
          </div>
        )}

        {/* Waveform */}
        <div className="mb-4">
          <VoiceWaveform
            level={micLevel}
            bars={32}
            variant={isListening ? 'mic' : isSpeaking ? 'speaker' : 'idle'}
            width={280}
            height={80}
          />
        </div>

        {/* Animated listening bars */}
        <div className="mb-4">
          <ListeningBars micLevel={micLevel} active={isListening} />
        </div>

        {/* Orb + controls */}
        <VoiceOrb state={state} micLevel={micLevel}>
          <button
            onClick={handleToggle}
            disabled={isProcessing}
            className={`
              w-full h-full rounded-full flex items-center justify-center
              transition-all duration-200
              ${isListening
                ? 'bg-primary text-primary-foreground'
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
              <IconSpeaker className="h-10 w-10" />
            ) : (
              <IconMicFilled className="h-10 w-10" />
            )}
          </button>
        </VoiceOrb>

        {/* Status text + transcription */}
        <div className="mt-6 text-center space-y-2 max-w-lg">
          <p className="text-sm text-muted-foreground">
            {isListening
              ? settings.pushToTalk
                ? 'Release spacebar to send'
                : 'Speak naturally — auto-sends after 2s of silence'
              : isProcessing
                ? 'Generating response...'
                : isSpeaking
                  ? 'Tap orb or press Space to interrupt'
                  : responseText
                    ? 'Tap to continue'
                    : settings.pushToTalk
                      ? 'Hold Space to talk'
                      : 'Tap microphone to start'
            }
          </p>

          {/* Live transcription */}
          {(finalText || interimText) && (
            <div className="bg-muted/50 rounded-lg px-4 py-2 text-sm">
              {finalText && <span className="text-foreground">{finalText}</span>}
              {interimText && <span className="text-primary/60 italic">{interimText}</span>}
            </div>
          )}

          {/* Live response (when not showing full transcript) */}
          {responseText && !showTranscript && (
            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg px-4 py-2 text-sm text-left max-h-32 overflow-y-auto">
              {responseText}
            </div>
          )}
        </div>
      </div>

      {/* Error */}
      {errorMessage && (
        <div className="absolute bottom-4 left-4 right-4 bg-red-500/10 border border-red-500/20 text-red-600 rounded-xl p-3 text-sm text-center">
          {errorMessage}
        </div>
      )}

      {/* Footer hint */}
      <div className="px-4 py-2 text-center">
        <p className="text-xs text-muted-foreground/50">
          {conversation.length > 0
            ? `${conversation.length} exchange${conversation.length === 1 ? '' : 's'} in this session`
            : 'No conversation yet'}
          {' · '}
          Speak during AI response to interrupt
        </p>
      </div>
    </div>
  )
}

// ── Transcript Exchange ───────────────────────────────────────────────

function TranscriptExchange({ exchange }: { exchange: VoiceExchange }) {
  return (
    <div className="space-y-1">
      <div className="text-sm text-primary/80">
        <span className="font-medium text-xs text-muted-foreground">You</span>
        <p className="mt-0.5">{exchange.userText}</p>
      </div>
      <div className="text-sm text-emerald-600 dark:text-emerald-400">
        <span className="font-medium text-xs text-muted-foreground">Assistant</span>
        <p className="mt-0.5">{exchange.assistantText}</p>
      </div>
    </div>
  )
}
