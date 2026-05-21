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
  streamingStats?: {
    tokens: number
    timeElapsed: number
    tokensPerSecond: number
  }
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
  streamingStats,
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
      className="shrink-0 border-t border-border/40 bg-background/80 backdrop-blur-sm px-3 py-3 sm:px-4 sm:py-3"
      style={{ paddingBottom: 'max(0.75rem, env(safe-area-inset-bottom))' }}
    >
      <div className="mx-auto max-w-2xl space-y-2">
        {streamingStats && loading && (
          <div className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-muted/50 text-xs text-muted-foreground shadow-sm" role="status" aria-live="polite">
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
              <span>Generating</span>
            </div>
            <div className="flex items-center gap-2 font-mono" aria-live="off">
              <span>{streamingStats.tokens}t</span>
              <span>{streamingStats.timeElapsed}s</span>
              <span>{streamingStats.tokensPerSecond}/s</span>
            </div>
          </div>
        )}

        {images.length > 0 && (
          <div className="flex gap-2 flex-wrap">
            {images.map((img) => (
              <ImagePreview 
                key={img.id} 
                image={img} 
                onRemove={handleRemoveImage}
              />
            ))}
          </div>
        )}
        
        <div className="flex items-center justify-center">
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
            hasContent={hasContent}
          />
        </div>
      </div>
    </section>
  )
}
