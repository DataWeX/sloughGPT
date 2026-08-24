'use client'

import { forwardRef, memo, useImperativeHandle, useRef, useState, useEffect, useCallback, useMemo } from 'react'
import { ChatInput } from './../input/ChatInput'
import { ChatScreen } from './ChatScreen'
import type { ChatInputProps } from './../input/ChatInput'
import type { ChatMessage } from './../types'
import type { ToolCallEvent } from '@/lib/stream-chat-response'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { cn, IconChevronDown } from '@sloughgpt/strui'

export interface ChatAreaProps extends Pick<ChatInputProps, 'value' | 'onChange' | 'onSend' | 'images' | 'onStop' | 'onAudioTranscript' | 'onGeneratedImage' | 'onPDFAnalysis' | 'onPDFError' | 'onExecuteCommand'> {
  messages: ChatMessage[]
  loading: boolean
  sessionLoading?: boolean
  health: ApiHealthSnapshot
  suggestions?: { text: string; icon: string }[]
  toolEvents?: ToolCallEvent[]
  ragVerification?: {
    confidence: number
    is_verified: boolean
    hallucination_rate: number
    citations: string
    grounded_claims: number
    hallucinated_claims: number
  } | null
  onRefreshHealth: () => void
  onCopy: (text: string) => void
  onRegenerate?: (messageId: string) => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  searchQuery?: string
  onSuggestionClick?: (text: string) => void
  onAddImage?: (dataUrl: string) => void
  onRemoveImage?: (id: string) => void
  className?: string
  model?: string
  isBookmarked?: (id: string) => boolean
  onBookmark?: (messageId: string) => void
  onDelete?: (messageId: string) => void
  onSaveToKnowledge?: (messageId: string, content: string) => void
  collapsibleLength?: number
}

export interface ChatAreaRef {
  scrollToBottom: () => void
}

const NEAR_BOTTOM_THRESHOLD = 100

export const ChatArea = memo(forwardRef<ChatAreaRef, ChatAreaProps>(
  function ChatArea({
    messages,
    loading,
    sessionLoading,
    health,
    suggestions,
    toolEvents,
    ragVerification,
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
    isBookmarked,
    onBookmark,
    onDelete,
    onSaveToKnowledge,
    collapsibleLength,
    ...inputProps
  }, ref) {
    const scrollRef = useRef<HTMLDivElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const [isNearBottom, setIsNearBottom] = useState(true)
    const prevMessageCountRef = useRef(messages.length)
    const prevLastContentLenRef = useRef(0)

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

    // Auto-scroll on new messages AND during streaming when user is near bottom
    useEffect(() => {
      const lastMsg = messages[messages.length - 1]
      const lastContentLen = lastMsg?.content?.length ?? 0
      const contentGrew = lastContentLen > prevLastContentLenRef.current
      const msgAdded = messages.length > prevMessageCountRef.current

      if (isNearBottom && (msgAdded || contentGrew)) {
        scrollRef.current?.scrollIntoView({ behavior: msgAdded ? 'smooth' : 'auto' })
      }
      prevMessageCountRef.current = messages.length
      prevLastContentLenRef.current = lastContentLen
    }, [messages, isNearBottom])

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
            suggestions={suggestions}
            onRefreshHealth={onRefreshHealth}
            onCopy={onCopy}
            onRegenerate={onRegenerate}
            onThumbsUp={onThumbsUp}
            onThumbsDown={onThumbsDown}
            onEdit={onEdit}
            searchQuery={searchQuery}
            onSuggestionClick={onSuggestionClick}
            toolEvents={toolEvents}
            ragVerification={ragVerification}
            isBookmarked={isBookmarked}
            onBookmark={onBookmark}
            onDelete={onDelete}
            onSaveToKnowledge={onSaveToKnowledge}
            collapsibleLength={collapsibleLength}
          />

          {filteredMessages.length > 0 && !isNearBottom && (
            <button
              type="button"
              onClick={() => scrollRef.current?.scrollIntoView({ behavior: 'smooth' })}
              className="sticky bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full border bg-background/80 backdrop-blur-sm shadow-lg hover:bg-accent/50 transition-all"
              aria-label="Jump to latest messages"
            >
              <IconChevronDown className="h-3.5 w-3.5" />
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
))
