'use client'

import { useState, useEffect, useRef } from 'react'
import { IconTrash, IconX, IconPlus, IconChat } from '@/components/ui'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/button'
import { ConversationListItem, type Conversation } from './ConversationListItem'

interface ConversationDrawerProps {
  isOpen: boolean
  onClose: () => void
  currentConversationId: string
  onLoadConversation: (conversationId: string) => void
  onDeleteConversation: (conversationId: string) => void
  onNewChat: () => void
  conversations?: Conversation[]
  onConversationsLoad?: () => void
}

const STORAGE_KEY = 'sloughgpt_chat_conversations'

export function ConversationDrawer({
  isOpen,
  onClose,
  currentConversationId,
  onLoadConversation,
  onDeleteConversation,
  onNewChat,
  conversations: externalConversations,
  onConversationsLoad,
}: ConversationDrawerProps) {
  const [localConversations, setLocalConversations] = useState<Conversation[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const drawerRef = useRef<HTMLDivElement>(null)

  const conversations = externalConversations || localConversations

  useEffect(() => {
    if (externalConversations && onConversationsLoad) {
      onConversationsLoad()
    } else if (!externalConversations) {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) setLocalConversations(JSON.parse(stored))
    }
  }, [currentConversationId, externalConversations, onConversationsLoad])

  const filteredConversations = conversations.filter(c =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.messages?.some(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const handleDelete = (conversationId: string) => {
    if (deleteConfirm === conversationId) {
      onDeleteConversation(conversationId)
      if (!externalConversations) setLocalConversations(prev => prev.filter(c => c.id !== conversationId))
      setDeleteConfirm(null)
    } else {
      setDeleteConfirm(conversationId)
      setTimeout(() => setDeleteConfirm(null), 3000)
    }
  }

  const handleClearAll = () => {
    if (confirm('Clear all conversations?')) {
      localStorage.removeItem(STORAGE_KEY)
      if (!externalConversations) setLocalConversations([])
      onNewChat()
    }
  }

  const handleExportChat = (conversation: Conversation, format: 'md' | 'json') => {
    let content: string
    let filename: string
    let mimeType: string

    if (format === 'md') {
      content = `# ${conversation.name}\n\n${(conversation.messages ?? []).map(m => {
        const role = m.role === 'user' ? '**User**' : '**Assistant**'
        return `${role}:\n${m.content}\n`
      }).join('\n')}`
      filename = `${conversation.name.replace(/[^a-z0-9]/gi, '_')}.md`
      mimeType = 'text/markdown'
    } else {
      content = JSON.stringify({ name: conversation.name, messages: conversation.messages }, null, 2)
      filename = `${conversation.name.replace(/[^a-z0-9]/gi, '_')}.json`
      mimeType = 'application/json'
    }

    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
      return () => {
        document.removeEventListener('keydown', handleEscape)
        document.body.style.overflow = ''
      }
    }
  }, [isOpen, onClose])

  return (
    <>
      <button
        onClick={onClose}
        className="fixed left-0 top-1/2 -translate-y-1/2 z-[25] hidden lg:hidden w-1.5 h-16 bg-primary/60 hover:bg-primary rounded-r-md transition-all duration-200"
        style={{ transform: 'translateY(-50%)', left: 0 }}
        aria-label="Open conversations"
      />

      {isOpen && (
        <div className="fixed inset-0 bg-black/40 z-40 lg:hidden" onClick={onClose} />
      )}

      <aside
        ref={drawerRef}
        className={cn(
          "fixed top-0 left-0 h-full w-72 sm:w-80",
          "bg-background border-r border-border z-50",
          "transform transition-transform duration-200 ease-out",
          "flex flex-col shadow-xl",
          "lg:hidden",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex items-center justify-between p-3 border-b border-border shrink-0">
          <h2 id="drawer-title" className="font-semibold text-sm">Conversations</h2>
          <div className="flex items-center gap-1">
            {conversations.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleClearAll}
                className="text-muted-foreground hover:text-destructive"
                title="Clear all conversations"
                aria-label="Clear all conversations"
              >
                <IconTrash className="h-3.5 w-3.5" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              aria-label="Close conversations"
            >
              <IconX className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="p-2 border-b border-border/50">
          <Button
            onClick={() => {
              onNewChat()
              onClose()
            }}
            className="w-full flex items-center justify-center gap-2"
            aria-label="Start new chat"
          >
            <IconPlus className="h-4 w-4" />
            New Chat
          </Button>
        </div>

        <div className="p-2 border-b border-border/30">
          <label htmlFor="conversation-search" className="sr-only">Search conversations</label>
          <Input
            id="conversation-search"
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="text-sm"
          />
        </div>

        <div 
          className="flex-1 overflow-y-auto p-2 space-y-1.5"
          role="listbox"
          aria-labelledby="drawer-title"
          aria-activedescendant={currentConversationId ? `conversation-${currentConversationId}` : undefined}
        >
          {filteredConversations.length === 0 ? (
            <div className="text-center py-10 px-4">
              <IconChat className="h-12 w-12 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">
                {searchQuery ? 'No matching conversations' : 'No conversations yet'}
              </p>
            </div>
          ) : (
            <div className="space-y-1.5">
              {filteredConversations.map((conversation) => (
                <ConversationListItem
                  key={conversation.id}
                  conversation={conversation}
                  isActive={conversation.id === currentConversationId}
                  onClick={() => {
                    if (conversation.id !== currentConversationId) {
                      onLoadConversation(conversation.id)
                      onClose()
                    }
                  }}
                  onDelete={() => handleDelete(conversation.id)}
                  onExport={(format) => handleExportChat(conversation, format)}
                />
              ))}
            </div>
          )}
        </div>

        <div className="p-3 border-t border-border text-xs text-muted-foreground text-center">
          {conversations.length} conversation{conversations.length !== 1 && 's'} • Local storage
        </div>
      </aside>
    </>
  )
}