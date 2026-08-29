'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconDownload, IconCopy, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ConversationExportProps {
  messages: ChatMessage[]
  sessionName?: string
  model?: string
  temperature?: number
  maxTokens?: number
  className?: string
}

interface ExportMetadata {
  exportedAt: string
  sessionName: string
  model: string
  temperature: number
  maxTokens: number
  messageCount: number
  totalCharacters: number
  totalWords: number
}

function computeMetadata(
  messages: ChatMessage[],
  sessionName: string,
  model: string,
  temperature: number,
  maxTokens: number,
): ExportMetadata {
  const totalCharacters = messages.reduce((sum, m) => sum + m.content.length, 0)
  const totalWords = messages.reduce((sum, m) => sum + m.content.split(/\s+/).filter(Boolean).length, 0)

  return {
    exportedAt: new Date().toISOString(),
    sessionName,
    model,
    temperature,
    maxTokens,
    messageCount: messages.length,
    totalCharacters,
    totalWords,
  }
}

function formatAsMarkdown(
  messages: ChatMessage[],
  metadata: ExportMetadata,
): string {
  const lines: string[] = []

  lines.push(`# ${metadata.sessionName || 'Chat Conversation'}`)
  lines.push('')
  lines.push('## Metadata')
  lines.push('')
  lines.push(`- **Exported:** ${metadata.exportedAt}`)
  lines.push(`- **Model:** ${metadata.model}`)
  lines.push(`- **Temperature:** ${metadata.temperature}`)
  lines.push(`- **Max Tokens:** ${metadata.maxTokens}`)
  lines.push(`- **Messages:** ${metadata.messageCount}`)
  lines.push(`- **Total Characters:** ${metadata.totalCharacters.toLocaleString()}`)
  lines.push(`- **Total Words:** ${metadata.totalWords.toLocaleString()}`)
  lines.push('')
  lines.push('---')
  lines.push('')
  lines.push('## Conversation')
  lines.push('')

  for (const msg of messages) {
    const role = msg.role === 'user' ? 'User' : 'Assistant'
    const timestamp = new Date(msg.timestamp).toLocaleString()
    lines.push(`### ${role} (${timestamp})`)
    lines.push('')
    lines.push(msg.content)
    lines.push('')
    if (msg.toolCalls && msg.toolCalls.length > 0) {
      lines.push('**Tool Calls:**')
      for (const tc of msg.toolCalls) {
        lines.push(`- ${tc.name} (${tc.status})`)
      }
      lines.push('')
    }
  }

  return lines.join('\n')
}

function formatAsJSON(
  messages: ChatMessage[],
  metadata: ExportMetadata,
): string {
  return JSON.stringify({
    metadata,
    messages: messages.map(m => ({
      role: m.role,
      content: m.content,
      timestamp: new Date(m.timestamp).toISOString(),
      toolCalls: m.toolCalls,
      images: m.images,
    })),
  }, null, 2)
}

export const ConversationExport = memo(function ConversationExport({
  messages,
  sessionName = 'Chat',
  model = 'unknown',
  temperature = 0.7,
  maxTokens = 1000,
  className,
}: ConversationExportProps) {
  const [format, setFormat] = useState<'markdown' | 'json'>('markdown')
  const [copied, setCopied] = useState(false)

  const metadata = computeMetadata(messages, sessionName, model, temperature, maxTokens)

  const getContent = useCallback(() => {
    if (format === 'markdown') {
      return formatAsMarkdown(messages, metadata)
    }
    return formatAsJSON(messages, metadata)
  }, [messages, metadata, format])

  const handleCopy = useCallback(async () => {
    const content = getContent()
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [getContent])

  const handleDownload = useCallback(() => {
    const content = getContent()
    const ext = format === 'markdown' ? 'md' : 'json'
    const mimeType = format === 'markdown' ? 'text/markdown' : 'application/json'
    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${sessionName.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.${ext}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [getContent, format, sessionName])

  if (messages.length === 0) {
    return (
      <div className={cn('text-xs text-muted-foreground text-center py-2', className)}>
        No messages to export
      </div>
    )
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {metadata.messageCount} messages · {metadata.totalWords.toLocaleString()} words
        </span>
        <div className="flex items-center gap-1">
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value as 'markdown' | 'json')}
            className="text-xs bg-transparent border border-border/30 rounded px-1.5 py-0.5"
          >
            <option value="markdown">Markdown</option>
            <option value="json">JSON</option>
          </select>
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-6 w-6"
            onClick={handleCopy}
            aria-label={copied ? 'Copied' : 'Copy to clipboard'}
          >
            {copied ? <IconCheck className="h-3 w-3" /> : <IconCopy className="h-3 w-3" />}
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-6 w-6"
            onClick={handleDownload}
            aria-label="Download conversation"
          >
            <IconDownload className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  )
})