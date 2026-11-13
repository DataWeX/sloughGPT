'use client'

import { memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { useConversationTags } from '@/features/chat/hooks/useConversationTags'
import { TagInput } from './TagInput'

interface ConversationTagsProps {
  sessionId: string
  className?: string
}

export const ConversationTags = memo(function ConversationTags({
  sessionId,
  className,
}: ConversationTagsProps) {
  const { getTags, addTag, removeTag, loading } = useConversationTags()

  if (loading) {
    return null
  }

  const tags = getTags(sessionId)

  return (
    <div className={cn('flex items-center', className)}>
      <TagInput
        tags={tags}
        onAdd={(tag) => addTag(sessionId, tag)}
        onRemove={(tag) => removeTag(sessionId, tag)}
        placeholder="Add tag..."
      />
    </div>
  )
})