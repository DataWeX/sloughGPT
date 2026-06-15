'use client'

import { forwardRef, useImperativeHandle, useRef, useState, useEffect, useCallback } from 'react'
import { ChatInput } from './ChatInput'
import { ChatScreen } from './ChatScreen'
import type { ChatInputProps } from './ChatInput'
import type { ChatMessage } from './types'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { cn } from '@/lib/cn'

export interface ChatAreaProps extends Pick<ChatInputProps, 'value' | 'onChange' | 'onSend' | 'images' | 'onStop' | 'onAudioTranscript' | 'onGeneratedImage' | 'onPDFAnalysis' | 'onPDFError'> {
  messages: ChatMessage[]
  loading: boolean
  sessionLoading?: boolean
  health: ApiHealthSnapshot
  onRefreshHealth: () => void
  onCopy: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  searchQuery?: string
  onSuggestionClick?: (text: string) => void
  onAddImage?: (dataUrl: string) => void
  onRemoveImage?: (id: string) => void
  className?: string
  model?: string
}

export interface ChatAreaRef {
  scrollToBottom: () => void
}

const NEAR_BOTTOM_THRESHOLD = 100

export const ChatArea = forwardRef<ChatAreaRef, ChatAreaProps>(
  function ChatArea({
    messages,
    loading,
    sessionLoading,
    health,
    onRefreshHealth,
    onCopy,
    onRegenerate,
    onThumbsUp,
    onThumbsDown,
    onEdit,
    searchQuery,
    onSuggestionClick,
    images,
    onAddImage,
    onRemoveImage,
    onAudioTranscript,
    onGeneratedImage,
    className,
    model,
    ...inputProps
  }, ref) {
    const scrollRef = useRef<HTMLDivElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const [isNearBottom, setIsNearBottom] = useState(true)
    const prevMessageCountRef = useRef(messages.length)

    const filteredMessages = searchQuery
      ? messages.filter(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
      : messages

    useImperativeHandle(ref, () => ({
      scrollToBottom: () => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
    }))

    const handleScroll = useCallback(() => {
      const el = containerRef.current
      if (!el) return
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
      setIsNearBottom(distFromBottom < NEAR_BOTTOM_THRESHOLD)
    }, [])

    // Auto-scroll on new messages when user is near bottom
    useEffect(() => {
      if (isNearBottom && messages.length > prevMessageCountRef.current) {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
      prevMessageCountRef.current = messages.length
    }, [messages.length, isNearBottom])

    // Scroll to bottom on initial load
    useEffect(() => {
      scrollRef.current?.scrollIntoView()
    }, [])

    return (
      <div className={cn("flex flex-col flex-1 min-h-0", className)}>
        <div
          ref={containerRef}
          className="flex-1 min-h-0 overflow-y-auto"
          onScroll={handleScroll}
          role="region"
          aria-label="Chat messages"
        >
          <ChatScreen
            ref={scrollRef}
            messages={filteredMessages}
            loading={loading}
            sessionLoading={sessionLoading}
            model={model}
            health={health}
            onRefreshHealth={onRefreshHealth}
            onCopy={onCopy}
            onRegenerate={onRegenerate}
            onThumbsUp={onThumbsUp}
            onThumbsDown={onThumbsDown}
            onEdit={onEdit}
            searchQuery={searchQuery}
            onSuggestionClick={onSuggestionClick}
          />

          {filteredMessages.length > 0 && !isNearBottom && (
            <button
              onClick={() => scrollRef.current?.scrollIntoView({ behavior: 'smooth' })}
              className="sticky bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full border bg-background/80 backdrop-blur-sm shadow-lg hover:bg-accent/50 transition-all"
              aria-label="Jump to latest messages"
            >
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
              {filteredMessages.length > 0 && (
                <span className="text-muted-foreground">{filteredMessages.length}</span>
              )}
            </button>
          )}
        </div>

        <ChatInput
          {...inputProps}
          loading={loading}
          health={health}
          images={images}
          onAddImage={onAddImage}
          onRemoveImage={onRemoveImage}
          onAudioTranscript={onAudioTranscript}
          onGeneratedImage={onGeneratedImage}
        />
      </div>
    )
  }
)
