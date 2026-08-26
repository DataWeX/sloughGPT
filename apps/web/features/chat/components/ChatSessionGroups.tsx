'use client'

import { useState, useCallback, useMemo, useEffect, memo } from 'react'
import { Button, IconX, IconPlus, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface SessionGroup {
  id: string
  name: string
  color: string
  sessionIds: string[]
  createdAt: number
}

interface Session {
  id: string
  title: string
}

interface ChatSessionGroupsProps {
  sessions: Session[]
  onAssignGroup: (sessionId: string, groupId: string | null) => void
  className?: string
}

const COLORS = [
  '#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
  '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#84cc16',
]

const STORAGE_KEY = 'chat-session-groups'

function loadGroups(): SessionGroup[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveGroups(groups: SessionGroup[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(groups))
}

export const ChatSessionGroups = memo(function ChatSessionGroups({
  sessions,
  onAssignGroup,
  className,
}: ChatSessionGroupsProps) {
  const [groups, setGroups] = useState<SessionGroup[]>([])
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newColor, setNewColor] = useState(COLORS[0])
  const [selectedColor, setSelectedColor] = useState<string | null>(null)
  const [assigningTo, setAssigningTo] = useState<string | null>(null)

  useEffect(() => {
    setGroups(loadGroups())
  }, [])

  const sessionsByGroup = useMemo(() => {
    const map: Record<string, Session[]> = {}
    const ungrouped: Session[] = []

    for (const session of sessions) {
      const group = groups.find(g => g.sessionIds.includes(session.id))
      if (group) {
        if (!map[group.id]) map[group.id] = []
        map[group.id].push(session)
      } else {
        ungrouped.push(session)
      }
    }

    return { map, ungrouped }
  }, [sessions, groups])

  const handleCreate = useCallback(() => {
    const trimmed = newName.trim()
    if (!trimmed) return

    const newGroup: SessionGroup = {
      id: crypto.randomUUID(),
      name: trimmed,
      color: newColor,
      sessionIds: [],
      createdAt: Date.now(),
    }

    const next = [...groups, newGroup]
    setGroups(next)
    saveGroups(next)
    setNewName('')
    setNewColor(COLORS[0])
    setCreating(false)
  }, [newName, newColor, groups])

  const handleDelete = useCallback((id: string) => {
    const next = groups.filter(g => g.id !== id)
    setGroups(next)
    saveGroups(next)
  }, [groups])

  const handleAssign = useCallback((sessionId: string, groupId: string) => {
    const next = groups.map(g => {
      if (g.id === groupId) {
        return { ...g, sessionIds: [...g.sessionIds, sessionId] }
      }
      return { ...g, sessionIds: g.sessionIds.filter(id => id !== sessionId) }
    })
    setGroups(next)
    saveGroups(next)
    onAssignGroup(sessionId, groupId)
    setAssigningTo(null)
  }, [groups, onAssignGroup])

  const handleUnassign = useCallback((sessionId: string, groupId: string) => {
    const next = groups.map(g => {
      if (g.id === groupId) {
        return { ...g, sessionIds: g.sessionIds.filter(id => id !== sessionId) }
      }
      return g
    })
    setGroups(next)
    saveGroups(next)
    onAssignGroup(sessionId, null)
  }, [groups, onAssignGroup])

  const filteredSessions = useMemo(() => {
    if (!selectedColor) return sessions
    const groupIds = groups.filter(g => g.color === selectedColor).map(g => g.id)
    return sessions.filter(s => {
      const group = groups.find(g => g.sessionIds.includes(s.id))
      return group && groupIds.includes(group.id)
    })
  }, [sessions, selectedColor, groups])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Session Groups</span>
        <Button
          variant="ghost"
          size="icon-sm"
          className="h-5 w-5"
          onClick={() => setCreating(!creating)}
          aria-label="Create group"
        >
          <IconPlus className="h-3 w-3" />
        </Button>
      </div>

      {creating && (
        <div className="p-2 border-b space-y-2">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Group name..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
          />
          <div className="flex gap-1 flex-wrap">
            {COLORS.map(color => (
              <button
                key={color}
                type="button"
                onClick={() => setNewColor(color)}
                className={cn(
                  'w-5 h-5 rounded-full border-2 transition-transform',
                  newColor === color ? 'border-foreground scale-110' : 'border-transparent',
                )}
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-6"
              onClick={handleCreate}
              disabled={!newName.trim()}
            >
              <IconCheck className="h-3 w-3 mr-1" />
              Create
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-6"
              onClick={() => setCreating(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      <div className="p-2 border-b flex gap-1 flex-wrap">
        <button
          type="button"
          onClick={() => setSelectedColor(null)}
          className={cn(
            'text-[10px] px-2 py-0.5 rounded transition-colors',
            selectedColor === null ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
          )}
        >
          All
        </button>
        {groups.map(group => (
          <button
            key={group.id}
            type="button"
            onClick={() => setSelectedColor(group.color)}
            className={cn(
              'text-[10px] px-2 py-0.5 rounded transition-colors flex items-center gap-1',
              selectedColor === group.color ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
            )}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: group.color }}
            />
            {group.name}
          </button>
        ))}
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {groups.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No groups yet. Create one to organize sessions.
          </p>
        ) : (
          <div className="divide-y">
            {groups.map(group => (
              <div key={group.id} className="px-3 py-2 hover:bg-muted/30 group">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: group.color }}
                    />
                    <span className="text-xs font-medium">{group.name}</span>
                    <span className="text-[10px] text-muted-foreground">
                      ({group.sessionIds.length})
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="h-5 w-5 opacity-0 group-hover:opacity-100"
                    onClick={() => handleDelete(group.id)}
                    title="Delete group"
                  >
                    <IconX className="h-3 w-3" />
                  </Button>
                </div>

                <div className="pl-4 space-y-1">
                  {(sessionsByGroup.map[group.id] || []).map(session => (
                    <div
                      key={session.id}
                      className="flex items-center justify-between text-xs"
                    >
                      <span className="truncate">{session.title}</span>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        className="h-4 w-4 opacity-0 group-hover:opacity-100"
                        onClick={() => handleUnassign(session.id, group.id)}
                        title="Remove from group"
                      >
                        <IconX className="h-2.5 w-2.5" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {sessionsByGroup.ungrouped.length > 0 && (
        <div className="p-2 border-t">
          <div className="text-[10px] text-muted-foreground mb-1">
            Ungrouped ({sessionsByGroup.ungrouped.length})
          </div>
          <div className="space-y-1 max-h-[100px] overflow-y-auto">
            {sessionsByGroup.ungrouped.map(session => (
              <div
                key={session.id}
                className="flex items-center justify-between text-xs"
              >
                <span className="truncate">{session.title}</span>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="h-4 w-4"
                  onClick={() => setAssigningTo(assigningTo === session.id ? null : session.id)}
                  title="Assign to group"
                >
                  <IconPlus className="h-2.5 w-2.5" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
})