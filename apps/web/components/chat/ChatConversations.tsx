'use client'

import { useState } from 'react'
import { ConversationListItem, type Conversation } from './ConversationListItem'
import { SectionHeader, SectionList } from '@/components/ui/section'

interface ChatConversationsProps {
  starredConversations: Conversation[]
  pinnedConversations: Conversation[]
  recentConversations: Conversation[]
  currentConversationId: string
  searchQuery?: string
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onStar?: (id: string, starred: boolean) => void
  onPin?: (id: string, pinned: boolean) => void
  onRename?: (id: string, newName: string) => void
  onDuplicate?: (id: string) => void
  onExport?: (conversation: Conversation, format: 'md' | 'json') => void
}

export function ChatConversations({
  starredConversations,
  pinnedConversations,
  recentConversations,
  currentConversationId,
  searchQuery,
  onSelect,
  onDelete,
  onStar,
  onPin,
  onRename,
  onDuplicate,
  onExport,
}: ChatConversationsProps) {
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const handleDelete = (conversationId: string) => {
    if (deleteConfirm === conversationId) {
      onDelete(conversationId)
      setDeleteConfirm(null)
    } else {
      setDeleteConfirm(conversationId)
    }
  }

  const hasAny = starredConversations.length > 0 || pinnedConversations.length > 0 || recentConversations.length > 0

  if (!hasAny) {
    return (
      <p className="text-xs text-muted-foreground text-center py-8 px-4">
        {searchQuery ? 'No matching conversations' : 'No conversations yet'}
      </p>
    )
  }

  return (
    <SectionList>
      {starredConversations.length > 0 && (
        <>
          <SectionHeader title="Starred" />
          {starredConversations.map((conversation) => (
            <ConversationListItem
              key={conversation.id}
              conversation={conversation}
              isActive={conversation.id === currentConversationId}
              onClick={() => onSelect(conversation.id)}
              onDelete={() => handleDelete(conversation.id)}
              onStar={(starred) => onStar?.(conversation.id, starred)}
              onPin={(pinned) => onPin?.(conversation.id, pinned)}
              onRename={(newName) => onRename?.(conversation.id, newName)}
              onDuplicate={() => onDuplicate?.(conversation.id)}
              onExport={(format) => onExport?.(conversation, format)}
              compact
            />
          ))}
        </>
      )}

      {pinnedConversations.length > 0 && (
        <>
          <SectionHeader title="Pinned" />
          {pinnedConversations.map((conversation) => (
            <ConversationListItem
              key={conversation.id}
              conversation={conversation}
              isActive={conversation.id === currentConversationId}
              onClick={() => onSelect(conversation.id)}
              onDelete={() => handleDelete(conversation.id)}
              onPin={(pinned) => onPin?.(conversation.id, pinned)}
              onRename={(newName) => onRename?.(conversation.id, newName)}
              onDuplicate={() => onDuplicate?.(conversation.id)}
              onExport={(format) => onExport?.(conversation, format)}
              compact
            />
          ))}
        </>
      )}

      {recentConversations.length > 0 && (
        <>
          <SectionHeader title="Recent" />
          {recentConversations.map((conversation) => (
            <ConversationListItem
              key={conversation.id}
              conversation={conversation}
              isActive={conversation.id === currentConversationId}
              onClick={() => onSelect(conversation.id)}
              onDelete={() => handleDelete(conversation.id)}
              onStar={(starred) => onStar?.(conversation.id, starred)}
              onPin={(pinned) => onPin?.(conversation.id, pinned)}
              onRename={(newName) => onRename?.(conversation.id, newName)}
              onDuplicate={() => onDuplicate?.(conversation.id)}
              onExport={(format) => onExport?.(conversation, format)}
              compact
            />
          ))}
        </>
      )}
    </SectionList>
  )
}