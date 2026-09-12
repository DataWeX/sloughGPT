'use client'

import { useState, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea } from '@sloughgpt/strui'
import { chatController } from '@/lib/chat-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('brainstorm')

const SUGGESTIONS = [
  'Name ideas',
  'Weekend plans',
  'Gift ideas',
  'Solve a problem',
  'Plan an event',
]

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function BrainstormPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const addToast = useToastStore((s) => s.addToast)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  const send = useCallback(async (text?: string) => {
    const message = text || input.trim()
    if (!message || isGenerating) return

    const userMsg: Message = { role: 'user', content: message }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsGenerating(true)

    try {
      const conversationHistory = [...messages, userMsg]
        .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
        .join('\n')

      const prompt = `You are a creative brainstorming partner. Respond with ideas in bullet points, not essays. Be concise and creative.\n\n${conversationHistory}\nAssistant:`

      let result = ''
      for await (const event of chatController.stream(prompt, { max_tokens: 500 })) {
        if (event.token) {
          result += event.token
          setMessages((prev) => {
            const updated = [...prev]
            const lastMsg = updated[updated.length - 1]
            if (lastMsg?.role === 'assistant') {
              lastMsg.content = result
            } else {
              updated.push({ role: 'assistant', content: result })
            }
            return updated
          })
        }
      }
    } catch (err) {
      _log.error('Brainstorm failed', { error: err instanceof Error ? err.message : String(err) })
      addToast('Brainstorming failed — is a model loaded?', 'error')
    } finally {
      setIsGenerating(false)
      scrollToBottom()
    }
  }, [input, messages, isGenerating, addToast, scrollToBottom])

  return (
    <PageContainer title="Brainstorm">
      <Card className="h-[calc(100vh-8rem)] flex flex-col">
        <CardContent className="flex-1 flex flex-col p-4 overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-3 mb-4">
            {messages.length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                <p className="text-lg mb-4">Let&apos;s think together. What&apos;s on your mind?</p>
                <div className="flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <Button
                      key={s}
                      variant="outline"
                      size="sm"
                      onClick={() => send(s)}
                      disabled={isGenerating}
                    >
                      {s}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                    msg.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="flex gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder="What's on your mind?"
              className="flex-1 resize-none text-sm"
              rows={2}
            />
            <Button
              onClick={() => send()}
              disabled={isGenerating || !input.trim()}
              className="self-end"
            >
              {isGenerating ? 'Thinking...' : 'Send'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </PageContainer>
  )
}
