'use client'

import { useState, useCallback, useEffect, useRef, memo } from 'react'
import { Button } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface SessionNote {
  id: string
  content: string
  createdAt: number
  updatedAt: number
}

interface ChatSessionNotesProps {
  sessionId: string
  className?: string
}

const STORAGE_KEY = 'chat-session-notes'

function loadNotes(sessionId: string): SessionNote[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const all: Record<string, SessionNote[]> = JSON.parse(raw)
    return all[sessionId] || []
  } catch {
    return []
  }
}

function saveNotes(sessionId: string, notes: SessionNote[]) {
  if (typeof window === 'undefined') return
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const all: Record<string, SessionNote[]> = raw ? JSON.parse(raw) : {}
    all[sessionId] = notes
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all))
  } catch {
    // ignore
  }
}

export const ChatSessionNotes = memo(function ChatSessionNotes({
  sessionId,
  className,
}: ChatSessionNotesProps) {
  const [notes, setNotes] = useState<SessionNote[]>([])
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [newNote, setNewNote] = useState('')
  const [showNew, setShowNew] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    setNotes(loadNotes(sessionId))
    setEditing(null)
    setNewNote('')
    setShowNew(false)
  }, [sessionId])

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus()
      textareaRef.current.selectionStart = textareaRef.current.value.length
    }
  }, [editing])

  const handleAdd = useCallback(() => {
    const trimmed = newNote.trim()
    if (!trimmed) return

    const note: SessionNote = {
      id: crypto.randomUUID(),
      content: trimmed,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }

    const next = [...notes, note]
    setNotes(next)
    saveNotes(sessionId, next)
    setNewNote('')
    setShowNew(false)
  }, [newNote, notes, sessionId])

  const handleEdit = useCallback((noteId: string) => {
    const note = notes.find(n => n.id === noteId)
    if (!note) return
    setEditing(noteId)
    setDraft(note.content)
  }, [notes])

  const handleSaveEdit = useCallback(() => {
    if (!editing) return
    const trimmed = draft.trim()
    if (!trimmed) return

    const next = notes.map(n =>
      n.id === editing
        ? { ...n, content: trimmed, updatedAt: Date.now() }
        : n
    )
    setNotes(next)
    saveNotes(sessionId, next)
    setEditing(null)
    setDraft('')
  }, [editing, draft, notes, sessionId])

  const handleCancelEdit = useCallback(() => {
    setEditing(null)
    setDraft('')
  }, [])

  const handleDelete = useCallback((noteId: string) => {
    const next = notes.filter(n => n.id !== noteId)
    setNotes(next)
    saveNotes(sessionId, next)
  }, [notes, sessionId])

  const formatDate = useCallback((timestamp: number) => {
    const date = new Date(timestamp)
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }, [])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Session Notes</span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">{notes.length}</span>
          <Button
            variant="ghost"
            size="sm"
            className="text-[10px] h-5"
            onClick={() => setShowNew(!showNew)}
          >
            {showNew ? 'Cancel' : '+ New'}
          </Button>
        </div>
      </div>

      {showNew && (
        <div className="p-2 border-b space-y-2">
          <textarea
            value={newNote}
            onChange={(e) => setNewNote(e.target.value)}
            placeholder="Add a note..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50 resize-none"
            rows={3}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleAdd()
              if (e.key === 'Escape') setShowNew(false)
            }}
          />
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-5"
              onClick={handleAdd}
              disabled={!newNote.trim()}
            >
              Save
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-5"
              onClick={() => setShowNew(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      <div className="max-h-[300px] overflow-y-auto">
        {notes.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No notes yet. Add a note to remember key points.
          </p>
        ) : (
          <div className="divide-y">
            {notes.map(note => (
              <div key={note.id} className="px-3 py-2 hover:bg-muted/30 group">
                {editing === note.id ? (
                  <div className="space-y-1">
                    <textarea
                      ref={textareaRef}
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50 resize-none"
                      rows={3}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSaveEdit()
                        if (e.key === 'Escape') handleCancelEdit()
                      }}
                    />
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-[10px] h-5"
                        onClick={handleSaveEdit}
                        disabled={!draft.trim()}
                      >
                        Save
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-[10px] h-5"
                        onClick={handleCancelEdit}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <p className="text-xs whitespace-pre-wrap">{note.content}</p>
                      <p className="text-[10px] text-muted-foreground mt-1">
                        {formatDate(note.updatedAt)}
                      </p>
                    </div>
                    <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100">
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        className="h-5 w-5"
                        onClick={() => handleEdit(note.id)}
                        title="Edit"
                      >
                        <span className="text-[10px]">✏</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        className="h-5 w-5"
                        onClick={() => handleDelete(note.id)}
                        title="Delete"
                      >
                        <span className="text-destructive text-[10px]">×</span>
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})