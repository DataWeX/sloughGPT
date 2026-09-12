'use client'

import { forwardRef, memo, useImperativeHandle, useMemo } from 'react'
import { ChatInput } from './../input/ChatInput'
import { ChatScreen } from './ChatScreen'
import type { ChatInputProps } from './../input/ChatInput'
import type { ChatMessage } from './../types'
import type { ToolCallEvent } from '@/lib/stream-chat-response'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { cn } from '@sloughgpt/strui'

export interface ChatAreaProps extends Pick<ChatInputProps, 'value' | 'onChange' | 'onSend' | 'images' | 'onStop' | 'onCancel' | 'onAudioRecorded' | 'onAudioTranscript' | 'onGeneratedImage' | 'onPDFAnalysis' | 'onPDFError' | 'onExecuteCommand'> {
  messages: ChatMessage[]
  loading: boolean
  sessionLoading?: boolean
  health: ApiHealthSnapshot
  suggestions?: { text: string; icon: string }[]
  toolEvents?: ToolCallEvent[]
  streamingStatus?: 'thinking' | 'generating' | 'tool_call' | 'context' | 'error'
  streamingToolName?: string
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
  onRegenerate?: (fromMessageId?: string) => void
  onRegenerateWithOptions?: (messageId: string, options: { temperature?: number; maxTokens?: number }) => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  onReact?: (messageId: string, emoji: string) => void
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
  temperature?: number
  contextLayers?: Array<{ type: 'knowledge' | 'memory' | 'rag' | 'tool' | 'soul' | 'system'; label: string; detail?: string }>
  noteMap?: Record<string, string>
  onAddNote?: (messageId: string) => void
  onPin?: (messageId: string) => void
  selectionMode?: boolean
  selectedMessageIds?: Set<string>
  onToggleSelection?: (messageId: string) => void
  hasThread?: (id: string) => boolean
  onThread?: (messageId: string) => void
  onQuickReply?: (messageId: string) => void
}

export interface ChatAreaRef {
  scrollToBottom: () => void
}

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
    onRegenerateWithOptions,
    onThumbsUp,
    onThumbsDown,
    onEdit,
    onReact,
    searchQuery,
    onSuggestionClick,
    images,
    onAddImage,
    onRemoveImage,
    onAudioRecorded,
    onAudioTranscript,
    onGeneratedImage,
    className,
    model,
    isBookmarked,
    onBookmark,
    onDelete,
    onSaveToKnowledge,
    collapsibleLength,
    temperature,
    contextLayers,
    noteMap,
    onAddNote,
    onPin,
    selectionMode,
    selectedMessageIds,
    onToggleSelection,
    hasThread,
    onThread,
    streamingStatus,
    streamingToolName,
    ...inputProps
  }, ref) {
    const filteredMessages = useMemo(() => searchQuery
      ? messages.filter(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
      : messages, [messages, searchQuery])

    // Virtuoso handles scroll via followOutput="smooth" — no manual scroll management needed.
    // Expose a no-op scrollToBottom for backward compatibility.
    useImperativeHandle(ref, () => ({
      scrollToBottom: () => {}
    }))

    return (
      <div className={cn("flex flex-col flex-1 min-h-0", className)}>
        <ChatScreen
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
          onReact={onReact}
          onPin={onPin}
          searchQuery={searchQuery}
          onSuggestionClick={onSuggestionClick}
          toolEvents={toolEvents}
          ragVerification={ragVerification}
          isBookmarked={isBookmarked}
          onBookmark={onBookmark}
          onDelete={onDelete}
          onSaveToKnowledge={onSaveToKnowledge}
          collapsibleLength={collapsibleLength}
          temperature={temperature}
          contextLayers={contextLayers}
          noteMap={noteMap}
          onAddNote={onAddNote}
          selectionMode={selectionMode}
          selectedMessageIds={selectedMessageIds}
          onToggleSelection={onToggleSelection}
          hasThread={hasThread}
          onThread={onThread}
        />

        <ChatInput
          {...inputProps}
          loading={loading}
          streamingStatus={streamingStatus}
          streamingToolName={streamingToolName}
          health={health}
          images={images}
          onAddImage={onAddImage}
          onRemoveImage={onRemoveImage}
          onAudioRecorded={onAudioRecorded}
          onAudioTranscript={onAudioTranscript}
          onGeneratedImage={onGeneratedImage}
        />
      </div>
    )
  }
))
