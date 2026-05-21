'use client'

import { ConversationListItem, type Conversation } from './ConversationListItem'

interface ChatConversationListProps {
  conversations: Conversation[]
  currentConversationId: string
  onLoadConversation: (conversationId: string) => void
  onDeleteConversation: (conversationId: string) => void
  onStarConversation?: (conversationId: string, starred: boolean) => void
  onRenameConversation?: (conversationId: string, newName: string) => void
  onDuplicateConversation?: (conversationId: string) => void
  onExport?: (conversation: Conversation, format: 'md' | 'json') => void
  searchQuery?: string
}

export function ChatConversationList({
  conversations,
  currentConversationId,
  onLoadConversation,
  onDeleteConversation,
  onStarConversation,
  onRenameConversation,
  onDuplicateConversation,
  onExport,
  searchQuery = '',
}: ChatConversationListProps) {
  const starred = conversations
    .filter(c => c.starred && c.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => new Date(b.updated_at ?? '').getTime() - new Date(a.updated_at ?? '').getTime())

  const pinned = conversations
    .filter(c => c.pinned && c.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => new Date(b.updated_at ?? '').getTime() - new Date(a.updated_at ?? '').getTime())

  const recent = conversations
    .filter(c => !c.pinned && c.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => new Date(b.updated_at ?? '').getTime() - new Date(a.updated_at ?? '').getTime())

  if (starred.length === 0 && pinned.length === 0 && recent.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        <p className="text-xs text-muted-foreground text-center py-8 px-4">
          {searchQuery ? 'No matching conversations' : 'No conversations yet'}
        </p>
      </div>
    )
  }

  const renderSection = (title: string, items: Conversation[]) => {
    if (items.length === 0) return null
    return (
      <div key={title}>
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-1 pt-2 pb-1">
          {title}
        </p>
        {items.map((conversation) => (
          <ConversationListItem
            key={conversation.id}
            conversation={conversation}
            isActive={conversation.id === currentConversationId}
            onClick={() => onLoadConversation(conversation.id)}
            onDelete={() => onDeleteConversation(conversation.id)}
            onStar={(starred) => onStarConversation?.(conversation.id, starred)}
            onRename={(newName) => onRenameConversation?.(conversation.id, newName)}
            onDuplicate={() => onDuplicateConversation?.(conversation.id)}
            onExport={(format) => onExport?.(conversation, format)}
            compact
          />
        ))}
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto overscroll-contain px-2 pb-2 space-y-1.5">
      {renderSection('Starred', starred)}
      {renderSection('Pinned', pinned)}
      {renderSection('Recent', recent)}
    </div>
  )
}