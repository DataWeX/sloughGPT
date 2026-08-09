'use client'

import { useState, useCallback, useRef } from 'react'
import type { ChatMessage } from '@/lib/chat-utils'
import { useToastStore } from '@/lib/toast-store'
import { imagesController } from '@/lib/images-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import type { ImageStyle } from '@/lib/images-controller'
import type { ChatMode } from '@/features/chat/components/toolbar/ModeBar'

const MODE_CONFIGS: Record<ChatMode, { placeholder: string }> = {
  chat: { placeholder: 'Type a message...' },
  write: { placeholder: 'What do you want to write about?' },
  decide: { placeholder: 'What do you need help deciding?' },
  explain: { placeholder: 'What do you want explained?' },
  translate: { placeholder: 'Text to translate...' },
  brainstorm: { placeholder: 'What should we brainstorm?' },
  wellness: { placeholder: 'How can I help you feel calm?' },
  create: { placeholder: 'Describe the image...' },
  read: { placeholder: 'Ask about your file...' },
  talk: { placeholder: 'Speak to me...' },
}

interface UseChatModeOptions {
  chat: {
    input: string
    setInput: React.Dispatch<React.SetStateAction<string>>
    sendMessage: (text?: string) => Promise<void>
    setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>
    setLoading: (loading: boolean) => void
  }
}

export function useChatMode({ chat }: UseChatModeOptions) {
  const [chatMode, setChatMode] = useState<ChatMode>('chat')
  const [writeTone, setWriteTone] = useState('Friendly')
  const [writeType, setWriteType] = useState('Email')
  const [decideStructure, setDecideStructure] = useState('Pros & Cons')
  const [explainDifficulty, setExplainDifficulty] = useState('Simple')
  const [translateLangPair, setTranslateLangPair] = useState('EN→ES')
  const [brainstormTopic, setBrainstormTopic] = useState('Name Ideas')
  const [wellnessType, setWellnessType] = useState('Sleep Story')
  const [createStyle, setCreateStyle] = useState('Realistic')
  const createAbortRef = useRef<AbortController | null>(null)

  const placeholder = MODE_CONFIGS[chatMode]?.placeholder || 'Type a message...'

  const handleCreateImage = useCallback(async (prompt: string) => {
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: prompt,
      timestamp: new Date(),
    }
    const pendingId = (Date.now() + 1).toString()
    const pendingMsg: ChatMessage = {
      id: pendingId,
      role: 'assistant',
      content: '✨ **Creating your image...**',
      timestamp: new Date(),
    }
    chat.setMessages(prev => [...prev, userMsg, pendingMsg])
    chat.setLoading(true)
    try {
      const result = await imagesController.generate(prompt, createStyle.toLowerCase() as ImageStyle)
      chat.setMessages(prev => prev.map(m =>
        m.id === pendingId
          ? { ...m, content: `Here's your ${createStyle.toLowerCase()} image:\n\n![${prompt}](${result.image})` }
          : m
      ))
    } catch (err: unknown) {
      chat.setMessages(prev => prev.map(m =>
        m.id === pendingId
          ? { ...m, content: `❌ Sorry, I couldn't create that image. ${extractErrorMessage(err, 'Please try again.')}` }
          : m
      ))
    } finally {
      chat.setLoading(false)
    }
  }, [chat, createStyle])

  const buildModePrompt = useCallback((input: string): string | null => {
    switch (chatMode) {
      case 'chat':
        return null // no transform
      case 'write':
        return `Write a ${writeTone.toLowerCase()} ${writeType.toLowerCase()} about: ${input}`
      case 'decide':
        return `Help me decide using ${decideStructure.toLowerCase()}: ${input}`
      case 'explain':
        return `Explain this at a ${explainDifficulty.toLowerCase()} level (as if explaining to a ${explainDifficulty.toLowerCase()} learner): ${input}`
      case 'translate': {
        const [src, tgt] = translateLangPair.split('→')
        return `Translate this from ${src} to ${tgt}: ${input}`
      }
      case 'brainstorm':
        return `Let's brainstorm ${brainstormTopic.toLowerCase()}. Be creative, give me ideas in a friendly list format: ${input}`
      case 'wellness': {
        const prompts: Record<string, string> = {
          'Sleep Story': 'Tell me a calming sleep story',
          'Meditation': 'Guide me through a short meditation',
          'Breathing': 'Guide me through a breathing exercise',
          'Affirmation': 'Share a positive affirmation',
        }
        return `Respond in a gentle, soothing tone. ${prompts[wellnessType] || 'Help me feel calm'}: ${input}`
      }
      case 'talk':
        return null
      default:
        return null
    }
  }, [chatMode, writeTone, writeType, decideStructure, explainDifficulty, translateLangPair, brainstormTopic, wellnessType])

  const handleSend = useCallback(async (readFileData?: { text: string; filename: string } | null) => {
    const input = chat.input.trim()

    if (chatMode === 'read') {
      if (!readFileData) {
        useToastStore.getState().addToast('Upload a file first, then ask your question', 'info')
        return
      }
      chat.setInput('')
      await chat.sendMessage(
        `[I'm asking about the file "${readFileData.filename}"]\n\nHere is the file content:\n${readFileData.text.slice(0, 12000)}\n\n---\n\nMy question: ${input}`
      )
      return
    }

    if (chatMode === 'create') {
      chat.setInput('')
      await handleCreateImage(input)
      return
    }

    if (chatMode === 'talk') {
      // talk mode handled by VoiceChatMode overlay
      return
    }

    const prompt = buildModePrompt(input)
    if (prompt) {
      chat.setInput('')
      await chat.sendMessage(prompt)
    } else {
      // 'chat' mode — send as-is
      await chat.sendMessage()
    }
  }, [chatMode, chat, buildModePrompt, handleCreateImage])

  return {
    chatMode,
    setChatMode,
    writeTone,
    setWriteTone,
    writeType,
    setWriteType,
    decideStructure,
    setDecideStructure,
    explainDifficulty,
    setExplainDifficulty,
    translateLangPair,
    setTranslateLangPair,
    brainstormTopic,
    setBrainstormTopic,
    wellnessType,
    setWellnessType,
    createStyle,
    setCreateStyle,
    placeholder,
    handleSend,
  }
}
