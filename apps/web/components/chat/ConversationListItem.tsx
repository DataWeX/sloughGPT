'use client'

import { Button } from '@sloughgpt/strui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@sloughgpt/strui'
import { IconStar, IconPin, IconChat, IconMore, IconEdit, IconCopy, IconDownload, IconTrash } from '@sloughgpt/strui'
import { cn } from '@/lib/cn'
import type { Conversation as ApiConversation } from '@/lib/session-controller'

export type Conversation = ApiConversation

interface ConversationListItemProps {
  conversation: Conversation
  isActive: boolean
  onClick: () => void
  onDelete: () => void
  onExport?: (format: 'md' | 'json') => void
  onStar?: (starred: boolean) => void
  onPin?: (pinned: boolean) => void
  onRename?: (newName: string) => void
  onRenameConversation?: (conversationId: string, newName: string) => void
  onDuplicate?: () => void
  compact?: boolean
}

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  if (isNaN(date.getTime())) return ''
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m`
  if (diffHours < 24) return `${diffHours}h`
  if (diffDays < 7) return `${diffDays}d`
  return date.toLocaleDateString()
}

function truncateMessage(content: string, maxLen = 40): string {
  if (!content) return 'Empty conversation'
  const firstLine = content.split('\n')[0]
  return firstLine.length > maxLen ? firstLine.slice(0, maxLen) + '…' : firstLine
}

export function ConversationListItem({
  conversation,
  isActive,
  onClick,
  onDelete,
  onExport,
  onStar,
  onPin,
  onRename,
  onRenameConversation,
  onDuplicate,
  compact = false,
}: ConversationListItemProps) {

  return (
    <div
      role="listitem"
      className={cn(
        "group relative rounded-md transition-all cursor-pointer border border-border/40",
        "bg-card/50",
        isActive
          ? "bg-secondary/50 border-primary/30"
          : "hover:bg-secondary/30 hover:border-border/60"
      )}
      onClick={onClick}
    >
      <div className="flex items-start gap-2 p-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            {conversation.pinned && <IconPin className="h-3 w-3 text-primary shrink-0" />}
            {conversation.starred && <IconStar className="h-3 w-3 text-warning shrink-0" filled />}
            <p className="text-xs font-medium truncate text-foreground">{conversation.name}</p>
          </div>

          {!compact && (
            <p className="text-xs text-muted-foreground/70 mt-0.5 line-clamp-1">
              {truncateMessage(conversation.messages?.[conversation.messages.length - 1]?.content || '')}
            </p>
          )}

          <div className="flex items-center gap-1 mt-0.5">
            <IconChat className="h-3 w-3 text-muted-foreground/70 shrink-0" />
            <p className="text-xs text-muted-foreground/70">
              {conversation.messages?.length ?? 0} · {formatDate(conversation.updated_at)}
            </p>
          </div>
        </div>

        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                className="h-6 w-6 p-0 hover:bg-transparent"
                onClick={(e) => e.stopPropagation()}
              >
                <IconMore className="h-3 w-3 text-muted-foreground/70" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-20 text-xs">
                <DropdownMenuItem onSelect={() => {
                  const newName = window.prompt('Rename conversation:', conversation.name)
                  const trimmed = newName?.trim()
                  if (trimmed && trimmed !== conversation.name) {
                    if (onRenameConversation) {
                      onRenameConversation(conversation.id, trimmed)
                    } else if (onRename) {
                      onRename(trimmed)
                    }
                  }
                }} className="text-xs py-1.5">
                  <IconEdit className="mr-2 h-3 w-3" />
                  Rename
                </DropdownMenuItem>
              {onDuplicate && (
                <DropdownMenuItem onSelect={() => onDuplicate()} className="text-xs py-1.5">
                  <IconCopy className="mr-2 h-3 w-3" />
                  Duplicate
                </DropdownMenuItem>
              )}
              {onStar && (
                <DropdownMenuItem onSelect={() => onStar(!conversation.starred)} className="text-xs py-1.5">
                  <IconStar className="mr-2 h-3 w-3" filled={conversation.starred} />
                  {conversation.starred ? 'Unstar' : 'Star'}
                </DropdownMenuItem>
              )}
              {onPin && (
                <DropdownMenuItem onSelect={() => onPin(!conversation.pinned)} className="text-xs py-1.5">
                  <IconPin className="mr-2 h-3 w-3" />
                  {conversation.pinned ? 'Unpin' : 'Pin to top'}
                </DropdownMenuItem>
              )}
              {onExport && (
                <>
                  <DropdownMenuItem onSelect={() => onExport('md')} className="text-xs py-1.5">
                    <IconDownload className="mr-2 h-3 w-3" />
                    Markdown
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => onExport('json')} className="text-xs py-1.5">
                    <IconDownload className="mr-2 h-3 w-3" />
                    JSON
                  </DropdownMenuItem>
                </>
              )}
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => {
                onDelete?.()
              }} className="text-destructive focus:text-destructive text-xs py-1">
                <IconTrash className="mr-2 h-3 w-3" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  )
}
