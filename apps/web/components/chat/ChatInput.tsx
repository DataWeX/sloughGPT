'use client'

import { useRef, useCallback } from 'react'
import { ImagePreview, type ImageAttachment } from './ImageUpload'
import { ChatInputRow } from './ChatInputRow'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'

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
}

export function ChatInput({ 
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
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = useCallback(() => {
    onSend()
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [onSend])

  const handleVoiceTranscript = useCallback((text: string) => {
    onChange(value ? `${value} ${text}` : text)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }, [value, onChange])

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
    ? 'API offline...' 
    : hasModel 
      ? 'Type a message...' 
      : 'Loading model...'
  const hasContent = value.trim().length > 0 || images.length > 0

  return (
    <section 
      className="shrink-0 bg-background/95 backdrop-blur-sm px-3 sm:px-4 pb-3"
      style={{ paddingBottom: 'max(0.75rem, env(safe-area-inset-bottom))' }}
    >
      <div className="mx-auto max-w-2xl">
        {loading && (
          <div className="flex justify-center pb-1.5" role="status" aria-live="polite">
            <div className="flex gap-[3px] items-center">
              <span className="w-1 h-1 rounded-full bg-primary/60 animate-bounce [animation-delay:0ms]" />
              <span className="w-1 h-1 rounded-full bg-primary/60 animate-bounce [animation-delay:150ms]" />
              <span className="w-1 h-1 rounded-full bg-primary/60 animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        {images.length > 0 && (
          <div className="flex gap-2 flex-wrap pb-2">
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
        />
      </div>
    </section>
  )
}
