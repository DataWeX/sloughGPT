'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'
import { toggleReaction, getReactions } from '@/lib/reaction-store'

export interface UseMessageActionsReturn {
  copied: boolean
  thumbsUp: boolean
  thumbsDown: boolean
  celebrate: boolean
  speaking: boolean
  savedToKnowledge: boolean
  showEmojiPicker: boolean
  showTranslateMenu: boolean
  showRetryOptions: boolean
  retryTemperature: number
  localReactions: Record<string, number>
  showDeleteConfirm: boolean
  addToast: (message: string, type?: string) => void
  copiedTimerRef: React.RefObject<ReturnType<typeof setTimeout> | null>
  celebrateTimerRef: React.RefObject<ReturnType<typeof setTimeout> | null>
  emojiPickerRef: React.RefObject<HTMLDivElement>
  translateMenuRef: React.RefObject<HTMLDivElement>
  retryOptionsRef: React.RefObject<HTMLDivElement>
  handleCopy: () => void
  handleThumbsUp: () => void
  handleThumbsDown: () => void
  handleSaveToKnowledge: () => void
  handleSpeak: () => void
  handleToggleReaction: (emoji: string) => void
  setRetryTemperature: (v: number) => void
  setShowRetryOptions: (v: boolean) => void
  setShowTranslateMenu: (v: boolean) => void
  setShowEmojiPicker: (v: boolean) => void
  setShowDeleteConfirm: (v: boolean) => void
}

export function useMessageActions(
  content: string,
  messageId: string,
  role?: 'user' | 'assistant',
  onCopy?: (text: string) => void,
  onThumbsUp?: (messageId: string) => void,
  onThumbsDown?: (messageId: string) => void,
  onSaveToKnowledge?: (messageId: string, content: string) => void,
  temperature: number = 0.7
): UseMessageActionsReturn {
  const [copied, setCopied] = useState(false)
  const [thumbsUp, setThumbsUp] = useState(false)
  const [thumbsDown, setThumbsDown] = useState(false)
  const [celebrate, setCelebrate] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [savedToKnowledge, setSavedToKnowledge] = useState(false)
  const [showEmojiPicker, setShowEmojiPicker] = useState(false)
  const [showTranslateMenu, setShowTranslateMenu] = useState(false)
  const [showRetryOptions, setShowRetryOptions] = useState(false)
  const [retryTemperature, setRetryTemperature] = useState(temperature)
  const [localReactions, setLocalReactions] = useState<Record<string, number>>({})
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const addToast = useToastStore(s => s.addToast) as (message: string, type?: string) => void
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const celebrateTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const emojiPickerRef = useRef<HTMLDivElement>(null!)
  const translateMenuRef = useRef<HTMLDivElement>(null!)
  const retryOptionsRef = useRef<HTMLDivElement>(null!)

  useEffect(() => {
    getReactions(messageId).then(reactions => {
      const counts: Record<string, number> = {}
      for (const [emoji, users] of Object.entries(reactions)) {
        counts[emoji] = users.length
      }
      setLocalReactions(counts)
    })
  }, [messageId])

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
      if (celebrateTimerRef.current) clearTimeout(celebrateTimerRef.current)
    }
  }, [])

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (emojiPickerRef.current && !emojiPickerRef.current.contains(e.target as Node)) {
        setShowEmojiPicker(false)
      }
      if (translateMenuRef.current && !translateMenuRef.current.contains(e.target as Node)) {
        setShowTranslateMenu(false)
      }
      if (retryOptionsRef.current && !retryOptionsRef.current.contains(e.target as Node)) {
        setShowRetryOptions(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true)
      onCopy?.(content)
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
      copiedTimerRef.current = setTimeout(() => setCopied(false), 2000)
    }).catch(() => {
      addToast('Failed to copy', 'error')
    })
  }, [content, onCopy, addToast])

  const handleThumbsUp = useCallback(() => {
    setThumbsUp(true)
    setThumbsDown(false)
    setCelebrate(true)
    if (celebrateTimerRef.current) clearTimeout(celebrateTimerRef.current)
    celebrateTimerRef.current = setTimeout(() => setCelebrate(false), 600)
    onThumbsUp?.(messageId)
  }, [messageId, onThumbsUp])

  const handleThumbsDown = useCallback(() => {
    setThumbsDown(true)
    setThumbsUp(false)
    onThumbsDown?.(messageId)
  }, [messageId, onThumbsDown])

  const handleSaveToKnowledge = useCallback(async () => {
    try {
      await knowledgeController.add(content, 'chat')
      setSavedToKnowledge(true)
      addToast('Saved to knowledge base', 'success')
    } catch {
      addToast('Failed to save to knowledge', 'error')
    }
  }, [content, addToast])

  const handleSpeak = useCallback(() => {
    if (speaking) {
      speechSynthesis.cancel()
      setSpeaking(false)
      return
    }
    const utterance = new SpeechSynthesisUtterance(content)
    utterance.onend = () => setSpeaking(false)
    utterance.onerror = () => setSpeaking(false)
    speechSynthesis.speak(utterance)
    setSpeaking(true)
  }, [content, speaking])

  const handleToggleReaction = useCallback(async (emoji: string) => {
    await toggleReaction(messageId, emoji)
    const reactions = await getReactions(messageId)
    const counts: Record<string, number> = {}
    for (const [e, users] of Object.entries(reactions)) {
      counts[e] = users.length
    }
    setLocalReactions(counts)
  }, [messageId])

  return {
    copied, thumbsUp, thumbsDown, celebrate, speaking, savedToKnowledge,
    showEmojiPicker, showTranslateMenu, showRetryOptions, retryTemperature,
    localReactions, showDeleteConfirm, addToast,
    copiedTimerRef, celebrateTimerRef, emojiPickerRef, translateMenuRef, retryOptionsRef,
    handleCopy, handleThumbsUp, handleThumbsDown, handleSaveToKnowledge, handleSpeak,
    handleToggleReaction, setRetryTemperature, setShowRetryOptions,
    setShowTranslateMenu, setShowEmojiPicker, setShowDeleteConfirm,
  }
}
