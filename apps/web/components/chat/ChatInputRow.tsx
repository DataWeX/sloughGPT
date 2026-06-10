'use client'

import type { RefObject } from 'react'
import { ChatInputAccessories } from './ChatInputAccessories'
import { ChatInputField } from './ChatInputField'
import { ChatSendButton } from './ChatSendButton'

interface ChatInputRowProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onStop?: () => void
  loading: boolean
  disabled: boolean
  placeholder: string
  textareaRef: RefObject<HTMLTextAreaElement>
  onImage: (dataUrl: string) => void
  onTranscript: (text: string) => void
  onAudioTranscript?: (text: string) => void
  onGeneratedImage?: (dataUrl: string, prompt: string) => void
  onPDFAnalysis?: (analysis: string, filename: string) => void
  onPDFError?: (error: string) => void
  hasContent: boolean
}

export function ChatInputRow({
  value, onChange, onSend, onStop,
  loading, disabled, placeholder,
  textareaRef, onImage, onTranscript, onAudioTranscript, onGeneratedImage,
  onPDFAnalysis, onPDFError, hasContent,
}: ChatInputRowProps) {
  return (
    <div className="flex items-end gap-1.5 w-full rounded-xl border border-border/40 bg-muted/10 px-2.5 py-1.5 focus-within:border-primary/30 focus-within:bg-muted/15 transition-all duration-200">
      <ChatInputAccessories
        onImage={onImage}
        onTranscript={onTranscript}
        disabled={disabled}
        onAudioTranscript={onAudioTranscript}
        onGeneratedImage={onGeneratedImage}
        onPDFAnalysis={onPDFAnalysis}
        onPDFError={onPDFError}
      />
      <ChatInputField
        value={value}
        onChange={onChange}
        onSend={onSend}
        placeholder={placeholder}
        disabled={disabled}
        textareaRef={textareaRef}
      />
      <ChatSendButton
        loading={loading}
        hasContent={hasContent}
        onSend={onSend}
        onStop={onStop}
        disabled={disabled}
      />
    </div>
  )
}
