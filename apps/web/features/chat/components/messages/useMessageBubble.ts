'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { ImageAttachment } from './../input/ImageUpload'
import type { AudioAttachment } from '@/lib/chat-utils'

export interface MessageBubbleState {
  isVisible: boolean
  isEditing: boolean
  bubbleRef: React.RefObject<HTMLDivElement | null>
  hasContent: boolean
  showActions: boolean
  id: string
}

export interface UseMessageBubbleProps {
  content: string
  role: 'user' | 'assistant'
  messageId?: string
  isStreaming?: boolean
  isError?: boolean
  onEdit?: (messageId: string, newContent: string) => void
}

export interface UseMessageBubbleReturn extends MessageBubbleState {
  handleEditStart: () => void
  handleEditCancel: () => void
}

export function useMessageBubble({
  content,
  role,
  messageId,
  isStreaming,
  isError,
  onEdit,
}: UseMessageBubbleProps): UseMessageBubbleReturn {
  const [isVisible, setIsVisible] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const bubbleRef = useRef<HTMLDivElement>(null)

  useEffect(() => { setIsVisible(true) }, [])

  const handleEditStart = useCallback(() => setIsEditing(true), [])
  const handleEditCancel = useCallback(() => setIsEditing(false), [])

  const hasContent = Boolean(content && content.trim().length > 0)
  const showActions = role === 'assistant' && hasContent && !isStreaming && !isError
  const id = messageId || 'msg'

  return {
    isVisible, isEditing, bubbleRef,
    hasContent, showActions, id,
    handleEditStart, handleEditCancel,
  }
}
