'use client'

import { memo } from 'react'
import { Button, IconX, IconPlus, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import { useSessionGroups, COLORS, type Session } from './useSessionGroups'

interface ChatSessionGroupsProps {
  sessions: Session[]
  onAssignGroup: (sessionId: string, groupId: string | null) => void
  className?: string
}

export const ChatSessionGroups = memo(function ChatSessionGroups({
  sessions,
  onAssignGroup,
  className,
}: ChatSessionGroupsProps) {
  const {
    groups, creating, newName, newColor, selectedColor, assigningTo,
    sessionsByGroup,
    setCreating, setNewName, setNewColor, setSelectedColor, setAssigningTo,
    handleCreate, handleDelete, handleAssign, handleUnassign,
  } = useSessionGroups(sessions, onAssignGroup)

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
                        onClick={() => handleUnassign(session.id)}
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
