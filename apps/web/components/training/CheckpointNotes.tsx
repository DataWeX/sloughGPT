'use client'

import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

const NOTES_KEY = 'sloughgpt-checkpoint-notes'

function loadNotes(): Record<string, string> {
  try {
    const raw = localStorage.getItem(NOTES_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function saveNotes(notes: Record<string, string>): void {
  localStorage.setItem(NOTES_KEY, JSON.stringify(notes))
}

interface CheckpointNotesProps {
  checkpoints: Checkpoint[]
}

export function CheckpointNotes({ checkpoints }: CheckpointNotesProps) {
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  useEffect(() => {
    setNotes(loadNotes())
  }, [])

  const handleSave = useCallback((name: string) => {
    const next = { ...notes, [name]: draft.trim() || '' }
    if (!draft.trim()) delete next[name]
    saveNotes(next)
    setNotes(next)
    setEditing(null)
  }, [notes, draft])

  const handleEdit = useCallback((name: string) => {
    setEditing(name)
    setDraft(notes[name] || '')
  }, [notes])

  const withNotes = checkpoints.filter(c => notes[c.name])

  if (checkpoints.length === 0) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Checkpoint notes</CardTitle>
        <span className="text-[10px] text-muted-foreground/50">
          {withNotes.length} annotated
        </span>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {checkpoints.map(c => {
            const hasNote = !!notes[c.name]
            const isEditing = editing === c.name
            return (
              <div key={c.name} className="flex items-start gap-2 py-1 border-b border-border/20 last:border-0">
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-medium truncate">{c.name}</p>
                  {isEditing ? (
                    <div className="mt-1 space-y-1">
                      <Textarea
                        value={draft}
                        onChange={e => setDraft(e.target.value)}
                        placeholder="Add a note..."
                        className="h-16 text-[11px] resize-none"
                      />
                      <div className="flex gap-1">
                        <Button size="sm" className="h-6 text-[10px]" onClick={() => handleSave(c.name)}>
                          Save
                        </Button>
                        <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setEditing(null)}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : hasNote ? (
                    <p className="text-[10px] text-muted-foreground/70 mt-0.5">{notes[c.name]}</p>
                  ) : null}
                </div>
                {!isEditing && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px] shrink-0"
                    onClick={() => handleEdit(c.name)}
                  >
                    {hasNote ? 'Edit' : 'Note'}
                  </Button>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
