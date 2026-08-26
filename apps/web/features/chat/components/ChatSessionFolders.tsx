'use client'

import { useState, useCallback, useMemo, useEffect, memo } from 'react'
import { Button, IconX, IconPlus, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface Folder {
  id: string
  name: string
  createdAt: number
}

interface Session {
  id: string
  title: string
  folderId: string | null
}

interface ChatSessionFoldersProps {
  sessions: Session[]
  onMoveSession: (sessionId: string, folderId: string | null) => void
  className?: string
}

const STORAGE_KEY = 'chat-folders'

function loadFolders(): Folder[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveFolders(folders: Folder[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(folders))
}

export const ChatSessionFolders = memo(function ChatSessionFolders({
  sessions,
  onMoveSession,
  className,
}: ChatSessionFoldersProps) {
  const [folders, setFolders] = useState<Folder[]>([])
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [dragOverFolder, setDragOverFolder] = useState<string | null>(null)

  useEffect(() => {
    setFolders(loadFolders())
  }, [])

  const sessionsByFolder = useMemo(() => {
    const map: Record<string, Session[]> = {}
    const unfiled: Session[] = []

    for (const session of sessions) {
      if (session.folderId) {
        if (!map[session.folderId]) map[session.folderId] = []
        map[session.folderId].push(session)
      } else {
        unfiled.push(session)
      }
    }

    return { map, unfiled }
  }, [sessions])

  const handleCreate = useCallback(() => {
    const trimmed = newName.trim()
    if (!trimmed) return

    const newFolder: Folder = {
      id: crypto.randomUUID(),
      name: trimmed,
      createdAt: Date.now(),
    }

    const next = [...folders, newFolder]
    setFolders(next)
    saveFolders(next)
    setNewName('')
    setCreating(false)
  }, [newName, folders])

  const handleDelete = useCallback((id: string) => {
    const next = folders.filter(f => f.id !== id)
    setFolders(next)
    saveFolders(next)
    for (const session of sessionsByFolder.map[id] || []) {
      onMoveSession(session.id, null)
    }
  }, [folders, sessionsByFolder, onMoveSession])

  const toggleFolder = useCallback((id: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent, folderId: string) => {
    e.preventDefault()
    setDragOverFolder(folderId)
  }, [])

  const handleDragLeave = useCallback(() => {
    setDragOverFolder(null)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent, folderId: string) => {
    e.preventDefault()
    const sessionId = e.dataTransfer.getData('text/plain')
    if (sessionId) {
      onMoveSession(sessionId, folderId)
    }
    setDragOverFolder(null)
  }, [onMoveSession])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Folders</span>
        <Button
          variant="ghost"
          size="icon-sm"
          className="h-5 w-5"
          onClick={() => setCreating(!creating)}
        >
          <IconPlus className="h-3 w-3" />
        </Button>
      </div>

      {creating && (
        <div className="p-2 border-b flex gap-1">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Folder name..."
            className="flex-1 text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
          />
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-6 w-6"
            onClick={handleCreate}
            disabled={!newName.trim()}
          >
            <IconCheck className="h-3 w-3" />
          </Button>
        </div>
      )}

      <div className="max-h-[400px] overflow-y-auto">
        {folders.length === 0 && sessionsByFolder.unfiled.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">No folders</p>
        ) : (
          <div className="divide-y">
            {folders.map(folder => {
              const folderSessions = sessionsByFolder.map[folder.id] || []
              const isExpanded = expandedFolders.has(folder.id)
              const isDragOver = dragOverFolder === folder.id

              return (
                <div
                  key={folder.id}
                  onDragOver={(e) => handleDragOver(e, folder.id)}
                  onDragLeave={handleDragLeave}
                  onDrop={(e) => handleDrop(e, folder.id)}
                  className={cn(isDragOver && 'bg-primary/10')}
                >
                  <div className="flex items-center gap-2 px-3 py-2 hover:bg-muted/30 group">
                    <button
                      type="button"
                      className="flex-1 text-left flex items-center gap-2"
                      onClick={() => toggleFolder(folder.id)}
                    >
                      <span className="text-xs">{isExpanded ? '▼' : '▶'}</span>
                      <span className="text-xs font-medium">{folder.name}</span>
                      <span className="text-[10px] text-muted-foreground">({folderSessions.length})</span>
                    </button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="h-5 w-5 opacity-0 group-hover:opacity-100"
                      onClick={() => handleDelete(folder.id)}
                      title="Delete folder"
                    >
                      <IconX className="h-3 w-3" />
                    </Button>
                  </div>

                  {isExpanded && folderSessions.length > 0 && (
                    <div className="pl-6 pb-2 space-y-1">
                      {folderSessions.map(session => (
                        <div
                          key={session.id}
                          draggable
                          onDragStart={(e) => e.dataTransfer.setData('text/plain', session.id)}
                          className="text-xs text-muted-foreground px-2 py-1 rounded hover:bg-muted/50 cursor-grab"
                        >
                          {session.title}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}

            {sessionsByFolder.unfiled.length > 0 && (
              <div
                onDragOver={(e) => handleDragOver(e, '__unfiled__')}
                onDragLeave={handleDragLeave}
                onDrop={(e) => handleDrop(e, '__unfiled__')}
                className={cn(dragOverFolder === '__unfiled__' && 'bg-primary/10')}
              >
                <div className="px-3 py-2">
                  <span className="text-[10px] text-muted-foreground">Unfiled ({sessionsByFolder.unfiled.length})</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
})