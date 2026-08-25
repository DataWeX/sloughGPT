'use client'

import { useState, useCallback } from 'react'
import { chatController } from '@/lib/chat-controller'
import type { ChatMessage } from '@/lib/chat-utils'

interface UseChatSummaryOptions {
  model: string
  temperature?: number
}

interface UseChatSummaryReturn {
  summary: string | null
  isGenerating: boolean
  error: string | null
  generateSummary: (messages: ChatMessage[]) => Promise<void>
  clearSummary: () => void
}

export function useChatSummary({
  model,
  temperature = 0.3,
}: UseChatSummaryOptions): UseChatSummaryReturn {
  const [summary, setSummary] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generateSummary = useCallback(async (messages: ChatMessage[]) => {
    if (messages.length === 0) {
      setError('No messages to summarize')
      return
    }

    setIsGenerating(true)
    setError(null)
    setSummary(null)

    try {
      const conversationText = messages
        .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
        .join('\n\n')

      const prompt = `Summarize the following conversation concisely. Focus on key topics, decisions, and outcomes. Use bullet points for main points.

Conversation:
${conversationText}

Summary:`

      const result = await chatController.send(prompt, {
        temperature,
        max_tokens: 500,
      })

      setSummary(result.message)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate summary')
    } finally {
      setIsGenerating(false)
    }
  }, [model, temperature])

  const clearSummary = useCallback(() => {
    setSummary(null)
    setError(null)
  }, [])

  return {
    summary,
    isGenerating,
    error,
    generateSummary,
    clearSummary,
  }
}