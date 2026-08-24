'use client'

import { useRef, useCallback, useEffect, memo } from 'react'
import { ImagePreview, type ImageAttachment } from './ImageUpload'
import { ChatInputRow } from './ChatInputRow'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import type { ChatCommand } from '@/lib/chat-commands'

export interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onStop?: () => void
  loading: boolean
  health: ApiHealthSnapshot
  images?: ImageAttachment[]
  onAddImage?: (dataUrl: string) => void
  onRemoveImage?: (id: string) => void
  onAudioTranscript?: (text: string) => void
  onGeneratedImage?: (dataUrl: string, prompt: string) => void
  onPDFAnalysis?: (analysis: string, filename: string) => void
  onPDFError?: (error: string) => void
  onExecuteCommand?: (cmd: ChatCommand, args: string[]) => void
}

export const ChatInput = memo(function ChatInput({
  value,
  onChange,
  onSend,
  onStop,
  loading,
  health,
  images = [],
  onAddImage,
  onRemoveImage,
  onAudioTranscript,
  onGeneratedImage,
  onPDFAnalysis,
  onPDFError,
  onExecuteCommand,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const pendingSendRef = useRef(false)

  const handleSend = useCallback(() => {
    onSend()
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [onSend])

  const handleVoiceTranscript = useCallback((text: string) => {
    onChange(value ? `${value} ${text}` : text)
    pendingSendRef.current = true
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }, [value, onChange])

  useEffect(() => {
    if (pendingSendRef.current && value.trim().length > 0) {
      pendingSendRef.current = false
      onSend()
      if (textareaRef.current) textareaRef.current.style.height = 'auto'
    } else if (pendingSendRef.current && value.trim().length === 0) {
      pendingSendRef.current = false
    }
  }, [value, onSend])

  const handleAddImage = useCallback((dataUrl: string) => {
    if (onAddImage) {
      onAddImage(dataUrl)
    }
  }, [onAddImage])

  const handleRemoveImage = useCallback((id: string) => {
    if (onRemoveImage) {
      onRemoveImage(id)
    }
  }, [onRemoveImage])

  const isDisabled = loading || health === 'offline'
  const hasModel = health !== null && health !== 'offline' && 'model_loaded' in health && health.model_loaded
  const placeholder = health === 'offline'
    ? 'Service offline...'
    : hasModel
      ? 'Type a message...'
      : 'Loading model...'
  const hasContent = value.trim().length > 0 || images.length > 0

  return (
    <section
      className="shrink-0 bg-background/95 backdrop-blur-sm px-4 sm:px-6 pb-2 pt-1"
      style={{ paddingBottom: 'max(0.5rem, env(safe-area-inset-bottom))' }}
    >
      <div className="mx-auto max-w-3xl">
        {loading && (
          <div className="flex justify-center pb-1" role="status" aria-live="polite" aria-atomic="true">
            <div className="flex gap-[3px] items-center">
              <span className="w-1 h-1 rounded-full bg-primary/60 animate-bounce [animation-delay:0ms]" />
              <span className="w-1 h-1 rounded-full bg-primary/60 animate-bounce [animation-delay:150ms]" />
              <span className="w-1 h-1 rounded-full bg-primary/60 animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        {images.length > 0 && (
          <div className="flex gap-2 flex-wrap pb-2" aria-label="Attached images">
            {images.map((img) => (
              <ImagePreview
                key={img.id}
                image={img}
                onRemove={handleRemoveImage}
              />
            ))}
          </div>
        )}

        <ChatInputRow
          value={value}
          onChange={onChange}
          onSend={handleSend}
          onStop={onStop}
          loading={loading}
          disabled={isDisabled}
          placeholder={placeholder}
          textareaRef={textareaRef}
          onImage={handleAddImage}
          onTranscript={handleVoiceTranscript}
          onAudioTranscript={onAudioTranscript}
          onGeneratedImage={onGeneratedImage}
          onPDFAnalysis={onPDFAnalysis}
          onPDFError={onPDFError}
          hasContent={hasContent}
          onExecuteCommand={onExecuteCommand}
        />
      </div>
    </section>
  )
})
