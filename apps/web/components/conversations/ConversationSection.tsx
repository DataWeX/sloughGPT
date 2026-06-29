'use client'

import ConversationRow from './ConversationRow'
import { type Conversation } from '@/lib/session-controller'

interface ConversationSectionProps {
  label: string
  conversations: Conversation[]
  selectedIds: Set<string>
  onToggleSelect: (id: string) => void
  onSelect: (id: string) => void
  onPin: (id: string, pinned: boolean) => void
  onStar: (id: string, starred: boolean) => void
  onArchive: (id: string, archived: boolean) => void
  onDelete: (id: string) => void
  onRename: (id: string) => void
  onExport: (conv: Conversation, format: 'md' | 'json') => void
}

export default function ConversationSection({
  label, conversations,
  selectedIds, onToggleSelect, onSelect, onPin, onStar, onArchive, onDelete, onRename, onExport,
}: ConversationSectionProps) {
  if (conversations.length === 0) return null

  return (
    <div>
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5 px-1">
        {label}
      </p>
      <div className="space-y-1">
        {conversations.map(c => (
          <ConversationRow
            key={c.id}
            conversation={c}
            selected={selectedIds.has(c.id)}
            onToggleSelect={() => onToggleSelect(c.id)}
            onSelect={() => onSelect(c.id)}
            onPin={(p) => onPin(c.id, p)}
            onStar={(s) => onStar(c.id, s)}
            onArchive={(a) => onArchive(c.id, a)}
            onDelete={() => onDelete(c.id)}
            onRename={() => onRename(c.id)}
            onExport={(f) => onExport(c, f)}
          />
        ))}
      </div>
    </div>
  )
}
