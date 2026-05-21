'use client'

import { ConversationListItem, type Conversation } from './ConversationListItem'

interface ConversationListProps {
  conversations: Conversation[]
  currentConversationId: string
  onLoadConversation: (conversationId: string) => void
  onDeleteConversation: (conversationId: string) => void
  onStarConversation?: (conversationId: string, starred: boolean) => void
  onPinConversation?: (conversationId: string, pinned: boolean) => void
  onRenameConversation?: (conversationId: string, newName: string) => void
  onDuplicateConversation?: (conversationId: string) => void
  onExport?: (conversation: Conversation, format: 'md' | 'json') => void
  searchQuery: string
}

export function ConversationList({
  conversations,
  currentConversationId,
  onLoadConversation,
  onDeleteConversation,
  onStarConversation,
  onPinConversation,
  onRenameConversation,
  onDuplicateConversation,
  onExport,
  searchQuery,
}: ConversationListProps) {
  const starredConversations = conversations
    .filter(c => c.starred && c.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => new Date(b.updated_at ?? '').getTime() - new Date(a.updated_at ?? '').getTime())

  const pinnedConversations = conversations
    .filter(c => c.pinned && c.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => new Date(b.updated_at ?? '').getTime() - new Date(a.updated_at ?? '').getTime())

  const recentConversations = conversations
    .filter(c => !c.pinned && c.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => new Date(b.updated_at ?? '').getTime() - new Date(a.updated_at ?? '').getTime())

  if (starredConversations.length === 0 && pinnedConversations.length === 0 && recentConversations.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        <p className="text-xs text-muted-foreground text-center py-8 px-4">
          {searchQuery ? 'No matching conversations' : 'No conversations yet'}
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto overscroll-contain">
      {starredConversations.length > 0 && (
        <div className="px-2 pt-2">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-1 mb-1.5">
            Starred
          </p>
          <div className="space-y-1.5">
            {starredConversations.map((conversation) => (
              <ConversationListItem
                key={conversation.id}
                conversation={conversation}
                isActive={conversation.id === currentConversationId}
                onClick={() => onLoadConversation(conversation.id)}
                onDelete={() => onDeleteConversation(conversation.id)}
                onStar={(starred) => onStarConversation?.(conversation.id, starred)}
                onPin={(pinned) => onPinConversation?.(conversation.id, pinned)}
                onRename={(newName) => onRenameConversation?.(conversation.id, newName)}
                onDuplicate={() => onDuplicateConversation?.(conversation.id)}
                onExport={(format) => onExport?.(conversation, format)}
                compact
              />
            ))}
          </div>
        </div>
      )}

      {pinnedConversations.length > 0 && (
        <div className="px-2 pt-2">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-1 mb-1.5">
            Pinned
          </p>
          <div className="space-y-1.5">
            {pinnedConversations.map((conversation) => (
              <ConversationListItem
                key={conversation.id}
                conversation={conversation}
                isActive={conversation.id === currentConversationId}
                onClick={() => onLoadConversation(conversation.id)}
                onDelete={() => onDeleteConversation(conversation.id)}
                onStar={(starred) => onStarConversation?.(conversation.id, starred)}
                onPin={(pinned) => onPinConversation?.(conversation.id, pinned)}
                onRename={(newName) => onRenameConversation?.(conversation.id, newName)}
                onDuplicate={() => onDuplicateConversation?.(conversation.id)}
                onExport={(format) => onExport?.(conversation, format)}
                compact
              />
            ))}
          </div>
        </div>
      )}

      {recentConversations.length > 0 && (
        <div className="px-2 pt-2 pb-2">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-1 mb-1.5">
            Recent
          </p>
          <div className="space-y-1.5">
            {recentConversations.map((conversation) => (
              <ConversationListItem
                key={conversation.id}
                conversation={conversation}
                isActive={conversation.id === currentConversationId}
                onClick={() => onLoadConversation(conversation.id)}
                onDelete={() => onDeleteConversation(conversation.id)}
                onStar={(starred) => onStarConversation?.(conversation.id, starred)}
                onPin={(pinned) => onPinConversation?.(conversation.id, pinned)}
                onRename={(newName) => onRenameConversation?.(conversation.id, newName)}
                onDuplicate={() => onDuplicateConversation?.(conversation.id)}
                onExport={(format) => onExport?.(conversation, format)}
                compact
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}