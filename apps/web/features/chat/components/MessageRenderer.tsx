'use client'

/**
 * MessageRenderer — single entry point for rendering chat messages.
 *
 * Handles: text streaming, markdown, code blocks, images, tool calls,
 * error states, and thinking indicators. All messages flow through this.
 */

import { memo, useMemo } from 'react'
import { cn } from '@sloughgpt/strui'
import { Markdown } from './messages/Markdown'
import { ToolCallPanel } from './messages/ToolCallPanel'
import { StreamingIndicator } from './StreamingIndicator'
import type { ToolCallEvent } from '@/lib/stream-chat-response'
import type { ImageAttachment } from './input/ImageUpload'

export interface RenderedMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date | string
  images?: ImageAttachment[]
  audio?: { url: string; durationMs: number }
  isError?: boolean
  isStreaming?: boolean
  toolCalls?: ToolCallEvent[]
  reasoning?: string
  metadata?: {
    model?: string
    tokens?: number
    latencyMs?: number
  }
}

interface MessageRendererProps {
  message: RenderedMessage
  searchQuery?: string
  isStreaming?: boolean
  className?: string
}

function ThinkingBlock() {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground py-1">
      <StreamingIndicator status="thinking" />
    </div>
  )
}

function StreamingCursor() {
  return (
    <span
      className="inline-block w-0.5 h-4 bg-primary ml-0.5 animate-pulse"
      aria-hidden="true"
    />
  )
}

function ImageGrid({ images }: { images: ImageAttachment[] }) {
  if (!images || images.length === 0) return null
  return (
    <div className="flex flex-wrap gap-2 mb-2">
      {images.map(img => (
        <div key={img.id} className="relative rounded-lg overflow-hidden border border-border/40 max-w-[240px]">
          <img
            src={img.dataUrl}
            alt={img.name}
            className="w-full h-auto max-h-[180px] object-cover"
            loading="lazy"
          />
          <span className="absolute bottom-0 left-0 right-0 text-[9px] text-muted-foreground bg-background/80 px-1.5 py-0.5 truncate">
            {img.name}
          </span>
        </div>
      ))}
    </div>
  )
}

function ToolCalls({ events }: { events: ToolCallEvent[] }) {
  if (!events || events.length === 0) return null
  return <ToolCallPanel events={events} />
}

function HighlightedContent({ content, query }: { content: string; query?: string }) {
  if (!query || !content) return <>{content}</>
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = content.split(new RegExp(`(${escaped})`, 'gi'))
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase()
          ? <mark key={i} className="bg-primary/20 rounded px-0.5">{part}</mark>
          : part
      )}
    </>
  )
}

export const MessageRenderer = memo(function MessageRenderer({
  message,
  searchQuery,
  isStreaming = false,
  className,
}: MessageRendererProps) {
  const { role, content, images, toolCalls, isError, reasoning, metadata } = message

  const renderedContent = useMemo(() => {
    if (!content && !isStreaming) return null

    // Empty thinking state
    if (!content && isStreaming && role === 'assistant') {
      return <ThinkingBlock />
    }

    // User messages — plain text with highlighting
    if (role === 'user') {
      return (
        <HighlightedContent content={content} query={searchQuery} />
      )
    }

    // Assistant messages — markdown with streaming cursor
    return (
      <div className="relative">
        <Markdown content={content} />
        {isStreaming && <StreamingCursor />}
      </div>
    )
  }, [content, role, isStreaming, searchQuery])

  return (
    <div className={cn('space-y-1', className)}>
      {/* Images */}
      <ImageGrid images={images || []} />

      {/* Tool calls */}
      {toolCalls && toolCalls.length > 0 && (
        <ToolCalls events={toolCalls} />
      )}

      {/* Reasoning (thinking) */}
      {reasoning && (
        <details className="text-xs text-muted-foreground border border-border/30 rounded-lg p-2">
          <summary className="cursor-pointer hover:text-foreground transition-colors">
            Reasoning
          </summary>
          <div className="mt-1.5 font-mono text-[11px] whitespace-pre-wrap opacity-70">
            {reasoning}
          </div>
        </details>
      )}

      {/* Main content */}
      {renderedContent}

      {/* Error indicator */}
      {isError && (
        <div className="text-xs text-destructive mt-1 flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-destructive" />
          Response interrupted
        </div>
      )}

      {/* Metadata (subtle) */}
      {metadata && (
        <div className="text-[9px] text-muted-foreground/40 font-mono mt-0.5">
          {metadata.model && <span>{metadata.model}</span>}
          {metadata.tokens != null && <span> · {metadata.tokens} tokens</span>}
          {metadata.latencyMs != null && <span> · {metadata.latencyMs}ms</span>}
        </div>
      )}
    </div>
  )
})
