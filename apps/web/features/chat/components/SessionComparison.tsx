'use client'

import { useState, useCallback, useMemo, memo } from 'react'
import { Button, Checkbox, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface Session {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  lastActivity: number
}

interface SessionComparisonProps {
  sessions: Session[]
  onClose: () => void
  className?: string
}

interface DiffResult {
  type: 'added' | 'removed' | 'unchanged' | 'modified'
  left?: ChatMessage
  right?: ChatMessage
}

function computeDiff(leftMessages: ChatMessage[], rightMessages: ChatMessage[]): DiffResult[] {
  const results: DiffResult[] = []
  const maxLen = Math.max(leftMessages.length, rightMessages.length)

  for (let i = 0; i < maxLen; i++) {
    const left = leftMessages[i]
    const right = rightMessages[i]

    if (!left) {
      results.push({ type: 'added', right })
    } else if (!right) {
      results.push({ type: 'removed', left })
    } else if (left.content === right.content) {
      results.push({ type: 'unchanged', left, right })
    } else {
      results.push({ type: 'modified', left, right })
    }
  }

  return results
}

export const SessionComparison = memo(function SessionComparison({
  sessions,
  onClose,
  className,
}: SessionComparisonProps) {
  const [leftId, setLeftId] = useState<string>(sessions[0]?.id || '')
  const [rightId, setRightId] = useState<string>(sessions[1]?.id || '')
  const [showUnchanged, setShowUnchanged] = useState(true)

  const leftSession = useMemo(() => sessions.find(s => s.id === leftId), [sessions, leftId])
  const rightSession = useMemo(() => sessions.find(s => s.id === rightId), [sessions, rightId])

  const diff = useMemo(() => {
    if (!leftSession || !rightSession) return []
    return computeDiff(leftSession.messages, rightSession.messages)
  }, [leftSession, rightSession])

  const filteredDiff = useMemo(() => {
    if (showUnchanged) return diff
    return diff.filter(d => d.type !== 'unchanged')
  }, [diff, showUnchanged])

  const stats = useMemo(() => {
    const added = diff.filter(d => d.type === 'added').length
    const removed = diff.filter(d => d.type === 'removed').length
    const modified = diff.filter(d => d.type === 'modified').length
    const unchanged = diff.filter(d => d.type === 'unchanged').length
    return { added, removed, modified, unchanged, total: diff.length }
  }, [diff])

  if (sessions.length < 2) {
    return (
      <div className={cn('border rounded-lg bg-card p-4', className)}>
        <p className="text-xs text-muted-foreground text-center">
          Need at least 2 sessions to compare
        </p>
      </div>
    )
  }

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Session Comparison</span>
        <Button variant="ghost" size="icon-sm" className="h-5 w-5" onClick={onClose} aria-label="Close comparison">
          <IconX className="h-3 w-3" />
        </Button>
      </div>

      <div className="flex gap-2 p-2 border-b">
        <select
          value={leftId}
          onChange={(e) => setLeftId(e.target.value)}
          className="flex-1 text-xs bg-transparent border rounded px-2 py-1"
        >
          {sessions.map(s => (
            <option key={s.id} value={s.id}>{s.title}</option>
          ))}
        </select>
        <span className="text-xs text-muted-foreground self-center">vs</span>
        <select
          value={rightId}
          onChange={(e) => setRightId(e.target.value)}
          className="flex-1 text-xs bg-transparent border rounded px-2 py-1"
        >
          {sessions.map(s => (
            <option key={s.id} value={s.id}>{s.title}</option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-3 px-3 py-2 border-b text-[10px]">
        <label className="flex items-center gap-1 cursor-pointer">
          <Checkbox
            checked={showUnchanged}
            onCheckedChange={(checked) => setShowUnchanged(checked === true)}
            className="rounded"
          />
          <span className="text-muted-foreground">Show unchanged</span>
        </label>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-success">+{stats.added}</span>
          <span className="text-destructive">-{stats.removed}</span>
          <span className="text-warning">~{stats.modified}</span>
          <span className="text-muted-foreground">={stats.unchanged}</span>
        </div>
      </div>

      <div className="max-h-[500px] overflow-y-auto">
        {filteredDiff.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            {showUnchanged ? 'Sessions are identical' : 'No differences found'}
          </p>
        ) : (
          <div className="divide-y">
            {filteredDiff.map((item, i) => (
              <DiffRow key={i} item={item} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
})

const DiffRow = memo(function DiffRow({ item, index }: { item: DiffResult; index: number }) {
  const bgClass = {
    added: 'bg-success/5',
    removed: 'bg-destructive/5',
    modified: 'bg-warning/5',
    unchanged: '',
  }[item.type]

  return (
    <div className={cn('flex text-xs', bgClass)}>
      <div className="w-1/2 border-r p-2 min-h-[40px]">
        {item.left ? (
          <div>
            <span className="text-[10px] text-muted-foreground mr-1">
              {item.left.role === 'user' ? 'You' : 'AI'}:
            </span>
            <span className={cn(item.type === 'removed' && 'line-through opacity-60')}>
              {item.left.content.slice(0, 100)}{item.left.content.length > 100 ? '…' : ''}
            </span>
          </div>
        ) : (
          <span className="text-muted-foreground/30">—</span>
        )}
      </div>
      <div className="w-1/2 p-2 min-h-[40px]">
        {item.right ? (
          <div>
            <span className="text-[10px] text-muted-foreground mr-1">
              {item.right.role === 'user' ? 'You' : 'AI'}:
            </span>
            <span className={cn(item.type === 'added' && 'font-medium')}>
              {item.right.content.slice(0, 100)}{item.right.content.length > 100 ? '…' : ''}
            </span>
          </div>
        ) : (
          <span className="text-muted-foreground/30">—</span>
        )}
      </div>
    </div>
  )
})