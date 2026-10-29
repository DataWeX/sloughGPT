'use client'

import { useState, useCallback } from 'react'
import { chatController } from '@/lib/chat-controller'
import type { ChatMessage } from '@/lib/chat-utils'

interface UseChatSummaryOptions {
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
  temperature = 0.3,
}: UseChatSummaryOptions = {}): UseChatSummaryReturn {
  const [summary, setSummary] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generateSummary = useCallback(
    async (messages: ChatMessage[]) => {
      if (messages.length === 0) {
        setError('No messages to summarize')
        return
      }

      setIsGenerating(true)
      setError(null)
      setSummary(null)

      try {
        const conversationText = messages
          .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
          .join('\n\n')

        const prompt = `Summarize the following conversation concisely. Focus on key topics, decisions, and outcomes. Use bullet points for main points.

Conversation:
${conversationText}

Summary:`

        // Summary rides the normal chat stream — it is a prompt, not a
        // feature: collect the tokens and fold at this edge. stream()'s
        // bracketed sentinels ([No model loaded], [Connection error…]) are
        // errors, not content.
        let out = ''
        for await (const token of chatController.stream(prompt, {
          temperature,
          max_tokens: 500,
        })) {
          out += token
        }

        if (out.startsWith('[') && out.endsWith(']')) {
          setError(out.slice(1, -1))
        } else {
          setSummary(out)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to generate summary')
      } finally {
        setIsGenerating(false)
      }
    },
    [temperature],
  )

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
