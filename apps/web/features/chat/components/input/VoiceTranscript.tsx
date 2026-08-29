'use client'

import { forwardRef } from 'react'
import type { VoiceExchange } from '@/features/chat/hooks/useVoiceChat'

interface VoiceTranscriptProps {
  conversation: VoiceExchange[]
  responseText: string
  isSpeaking: boolean
}

export const VoiceTranscript = forwardRef<HTMLDivElement, VoiceTranscriptProps>(
  function VoiceTranscript({ conversation, responseText, isSpeaking }, ref) {
    return (
      <div
        ref={ref}
        className="w-full max-w-lg mb-6 max-h-64 overflow-y-auto rounded-xl border border-border/50 bg-muted/30 p-3 space-y-3"
      >
        {conversation.map((ex) => (
          <TranscriptExchange key={ex.id} exchange={ex} />
        ))}
        {responseText && !isSpeaking && (
          <div className="text-sm text-emerald-600 dark:text-emerald-400">
            <span className="font-medium text-xs text-muted-foreground">Assistant</span>
            <p className="mt-0.5">{responseText}</p>
          </div>
        )}
      </div>
    )
  }
)

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
