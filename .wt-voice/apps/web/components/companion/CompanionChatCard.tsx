'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, cn } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

interface CompanionChatCardProps {
  onSend?: (message: string) => Promise<string>
}

export function CompanionChatCard({ onSend }: CompanionChatCardProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading || !onSend) return
    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}-u`,
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    try {
      const response = await onSend(userMsg.content)
      const aiMsg: ChatMessage = {
        id: `msg-${Date.now()}-a`,
        role: 'assistant',
        content: response,
        timestamp: Date.now(),
      }
      setMessages(prev => [...prev, aiMsg])
    } catch {
      const errMsg: ChatMessage = {
        id: `msg-${Date.now()}-e`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: Date.now(),
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setLoading(false)
    }
  }, [input, loading, onSend])

  return (
    <Card data-testid="companion-chat">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Chat</CardTitle>
          {messages.length > 0 && (
            <Button
              size="sm"
              variant="ghost"
              className="text-destructive text-[10px]"
              onClick={() => setMessages([])}
            >
              Clear
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64 overflow-y-auto space-y-2 mb-3 p-2 rounded bg-muted/30" data-testid="chat-messages">
          {messages.length === 0 && (
            <div className="text-xs text-muted-foreground text-center py-8">
              Start a conversation with your companion.
            </div>
          )}
          {messages.map(msg => (
            <div key={msg.id} className={cn('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
              <div
                className={cn(
                  'max-w-[80%] px-3 py-1.5 rounded-lg text-xs',
                  msg.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-foreground'
                )}
                data-testid={`message-${msg.role}`}
              >
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-muted text-foreground px-3 py-1.5 rounded-lg text-xs animate-pulse">
                Thinking...
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Say something..."
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            disabled={loading}
            data-testid="chat-input"
          />
          <Button onClick={handleSend} disabled={loading || !input.trim()}>
            Send
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
