'use client'

import { useState, useEffect } from 'react'
import { Button, IconX } from '@sloughgpt/strui'
import type { Card, CardNote } from './types'
import { COLUMN_LABELS } from './types'
import { HashChain } from './HashChain'
import { addCardNote, updateCardNote, deleteCardNote } from '@/lib/planner-client'

interface CardEditorProps {
  card: Card | null
  onClose: () => void
  onUpdate: (cardId: string, data: Partial<Card>) => void
  onDelete: (cardId: string) => void
}

const input =
  'w-full rounded-md bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600 transition-colors'
const label = 'text-xs font-medium text-zinc-400'

export function CardEditor({ card, onClose, onUpdate, onDelete }: CardEditorProps) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [column, setColumn] = useState('todo')
  const [priority, setPriority] = useState('medium')
  const [tags, setTags] = useState('')
  const [assignee, setAssignee] = useState('')
  const [dueDate, setDueDate] = useState('')
  const [sprint, setSprint] = useState('')
  const [gh, setGh] = useState('')
  const [notes, setNotes] = useState<CardNote[]>([])
  const [treeVersion, setTreeVersion] = useState(0)
  const [noteDraft, setNoteDraft] = useState('')
  const [noteAuthor, setNoteAuthor] = useState('')
  const [editingNote, setEditingNote] = useState<string | null>(null)
  const [editDraft, setEditDraft] = useState('')
  const [noteBusy, setNoteBusy] = useState(false)
  const [noteError, setNoteError] = useState<string | null>(null)

  const cardId = card?.id ?? null

  useEffect(() => {
    if (card) {
      setTitle(card.title)
      setDescription(card.description)
      setColumn(card.column)
      setPriority(card.priority)
      setTags(card.tags.join(', '))
      setAssignee(card.assignee)
      setDueDate(card.due_date)
      setSprint(card.sprint)
      setGh(card.gh)
    }
    // Form fields only re-init when switching cards — typing survives polls.
  }, [cardId]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (card) setNotes(card.notes)
  }, [card])

  if (!card) return null

  const handleSave = () => {
    onUpdate(card.id, {
      title: title.trim(),
      description: description.trim(),
      column,
      priority: priority as Card['priority'],
      tags: tags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean),
      assignee: assignee.trim(),
      due_date: dueDate.trim(),
      sprint: sprint.trim(),
      gh: gh.trim(),
    })
  }

  // Note mutations — server returns the updated card (notes + tree in one save).
  const withNotes = async (fn: () => Promise<{ card: Card }>) => {
    setNoteBusy(true)
    setNoteError(null)
    try {
      const { card: updated } = await fn()
      setNotes(updated.notes)
      setTreeVersion((v) => v + 1)
    } catch (e) {
      setNoteError(e instanceof Error ? e.message : 'note update failed')
    } finally {
      setNoteBusy(false)
    }
  }

  const handleAddNote = () => {
    if (!noteDraft.trim()) return
    const draft = noteDraft
    setNoteDraft('')
    void withNotes(() => addCardNote(card.id, draft, noteAuthor))
  }

  const handleSaveNote = (noteId: string) => {
    if (!editDraft.trim()) return
    const draft = editDraft
    setEditingNote(null)
    void withNotes(() => updateCardNote(card.id, noteId, draft))
  }

  const handleDeleteNote = (noteId: string) => {
    void withNotes(() => deleteCardNote(card.id, noteId))
  }

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end"
      role="dialog"
      aria-modal="true"
      aria-label="Edit card"
    >
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />
      <div className="relative w-full max-w-md bg-zinc-900 border-l border-zinc-800 shadow-2xl overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-zinc-900 border-b border-zinc-800 px-5 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-base font-medium text-zinc-200">Edit Card</h2>
            <p className="text-[10px] font-mono text-zinc-600 mt-0.5">{card.id}</p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close"
            className="text-zinc-500 hover:text-zinc-300"
          >
            <IconX className="h-4 w-4" />
          </Button>
        </div>

        <div className="p-5 space-y-4">
          <div>
            <label htmlFor="card-title" className={label}>
              Title
            </label>
            <input
              id="card-title"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className={`${input} mt-1`}
            />
          </div>
          <div>
            <label htmlFor="card-description" className={label}>
              Description
            </label>
            <textarea
              id="card-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
              className={`${input} mt-1 resize-none`}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="card-column" className={label}>
                Column
              </label>
              <select
                id="card-column"
                value={column}
                onChange={(e) => setColumn(e.target.value)}
                className={`${input} mt-1`}
              >
                {Object.entries(COLUMN_LABELS).map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="card-priority" className={label}>
                Priority
              </label>
              <select
                id="card-priority"
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                className={`${input} mt-1`}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
          </div>
          <div>
            <label htmlFor="card-tags" className={label}>
              Tags
            </label>
            <input
              id="card-tags"
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="comma, separated"
              className={`${input} mt-1`}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="card-assignee" className={label}>
                Assignee
              </label>
              <input
                id="card-assignee"
                type="text"
                value={assignee}
                onChange={(e) => setAssignee(e.target.value)}
                className={`${input} mt-1`}
              />
            </div>
            <div>
              <label htmlFor="card-due-date" className={label}>
                Due Date
              </label>
              <input
                id="card-due-date"
                type="text"
                value={dueDate}
                onChange={(e) => setDueDate(e.target.value)}
                className={`${input} mt-1`}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="card-sprint" className={label}>
                Sprint
              </label>
              <input
                id="card-sprint"
                type="text"
                value={sprint}
                onChange={(e) => setSprint(e.target.value)}
                className={`${input} mt-1`}
              />
            </div>
            <div>
              <label htmlFor="card-gh" className={label}>
                GitHub
              </label>
              <input
                id="card-gh"
                type="text"
                value={gh}
                onChange={(e) => setGh(e.target.value)}
                className={`${input} mt-1`}
              />
            </div>
          </div>

          {/* Actions */}
          <div
            className="flex justify-between pt-4"
            style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
          >
            <button
              onClick={() => onDelete(card.id)}
              className="px-3 py-1.5 rounded-md text-sm text-red-400 hover:text-red-300 hover:bg-red-950/50 transition-colors"
            >
              Delete
            </button>
            <div className="flex gap-2">
              <button
                onClick={onClose}
                className="px-3 py-1.5 rounded-md text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="px-3 py-1.5 rounded-md text-sm bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 transition-colors"
              >
                Save
              </button>
            </div>
          </div>

          {/* Notes — each mutation rehashes the chain */}
          <div className="pt-4 space-y-2" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <div className="flex items-center justify-between">
              <span className={label}>Notes</span>
              <span className="text-[10px] font-mono text-zinc-600">{notes.length}</span>
            </div>

            {notes.length > 0 && (
              <ul className="space-y-1.5">
                {notes.map((n) => (
                  <li
                    key={n.id}
                    className="rounded-md bg-zinc-950/60 border border-zinc-800 px-2.5 py-2 group"
                  >
                    {editingNote === n.id ? (
                      <div className="space-y-1.5">
                        <textarea
                          value={editDraft}
                          onChange={(e) => setEditDraft(e.target.value)}
                          rows={3}
                          autoFocus
                          className={`${input} resize-none`}
                          aria-label="Edit note"
                        />
                        <div className="flex justify-end gap-1.5">
                          <button
                            onClick={() => setEditingNote(null)}
                            className="px-2 py-1 rounded text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
                          >
                            Cancel
                          </button>
                          <button
                            onClick={() => handleSaveNote(n.id)}
                            disabled={noteBusy || !editDraft.trim()}
                            className="px-2 py-1 rounded text-[11px] bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 disabled:opacity-40 transition-colors"
                          >
                            Save
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p className="text-[13px] text-zinc-300 whitespace-pre-wrap break-words">
                          {n.text}
                        </p>
                        <div className="flex items-center justify-between mt-1.5">
                          <span className="text-[10px] font-mono text-zinc-600">
                            {n.author} · {n.created_at.slice(0, 10)}
                          </span>
                          <span className="flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                            <button
                              onClick={() => {
                                setEditingNote(n.id)
                                setEditDraft(n.text)
                              }}
                              className="px-1.5 py-0.5 rounded text-[10px] text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
                              aria-label="Edit note"
                            >
                              edit
                            </button>
                            <button
                              onClick={() => handleDeleteNote(n.id)}
                              disabled={noteBusy}
                              className="px-1.5 py-0.5 rounded text-[10px] text-zinc-600 hover:text-red-400 hover:bg-red-950/40 disabled:opacity-40 transition-colors"
                              aria-label="Delete note"
                            >
                              delete
                            </button>
                          </span>
                        </div>
                      </>
                    )}
                  </li>
                ))}
              </ul>
            )}

            <div className="space-y-1.5">
              <textarea
                value={noteDraft}
                onChange={(e) => setNoteDraft(e.target.value)}
                rows={2}
                placeholder="Add a note…"
                aria-label="New note"
                className={`${input} resize-none`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleAddNote()
                }}
              />
              <div className="flex items-center gap-1.5">
                <input
                  type="text"
                  value={noteAuthor}
                  onChange={(e) => setNoteAuthor(e.target.value)}
                  placeholder="author"
                  aria-label="Note author"
                  className="rounded-md bg-zinc-800 border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600 w-24 transition-colors"
                />
                <button
                  onClick={handleAddNote}
                  disabled={noteBusy || !noteDraft.trim()}
                  className="px-2.5 py-1 rounded-md text-[11px] bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 disabled:opacity-40 transition-colors"
                >
                  {noteBusy ? 'saving…' : 'Add note'}
                </button>
                <span className="text-[9px] font-mono text-zinc-700 ml-auto">⌘↵</span>
              </div>
              {noteError && (
                <p className="text-[11px] text-red-400 font-mono" role="alert">
                  {noteError}
                </p>
              )}
            </div>
          </div>

          {/* Hash Chain */}
          <HashChain cardId={card.id} refreshKey={treeVersion} />
        </div>
      </div>
    </div>
  )
}
