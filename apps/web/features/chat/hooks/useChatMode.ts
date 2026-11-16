'use client'
import { notify } from '@/lib/notify'

import { useCallback } from 'react'
import type { ChatMessage } from '@/lib/chat-utils'

interface UseChatModeOptions {
  chat: {
    input: string
    setInput: React.Dispatch<React.SetStateAction<string>>
    sendMessage: (text?: string) => Promise<void>
    setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>
    setLoading: (loading: boolean) => void
  }
}

// Plain composer: no prompt-decoration modes. The general agent infers intent
// from the raw text — the only transform left is the file-attach injection
// (a selected file travels with the question), which is always available and
// never mode-gated.
export function useChatMode({ chat }: UseChatModeOptions) {
  const handleSend = useCallback(
    async (readFileData?: { text: string; filename: string } | null) => {
      const input = chat.input.trim()

      if (readFileData) {
        if (!input) {
          notify.toast('Ask a question about your file', { type: 'info' })
          return
        }
        chat.setInput('')
        await chat.sendMessage(
          `[I'm asking about the file "${readFileData.filename}"]\n\nHere is the file content:\n${readFileData.text.slice(0, 12000)}\n\n---\n\nMy question: ${input}`,
        )
        return
      }

      // Plain send — no transform.
      await chat.sendMessage()
    },
    [chat],
  )

  return { handleSend }
}
