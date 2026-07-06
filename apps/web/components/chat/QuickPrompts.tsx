'use client'

import { useState, useCallback, useMemo } from 'react'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import type { QuickPrompt } from '@/lib/quick-prompts'
import { listPromptsByCategory, createPrompt, updatePrompt, deletePrompt, resetToDefaults, applyPrompt } from '@/lib/quick-prompts'

const CATEGORY_LABELS: Record<string, string> = {
  writing: 'Writing',
  coding: 'Coding',
  planning: 'Planning',
  learning: 'Learning',
  custom: 'Custom',
}

interface QuickPromptsProps {
  onUsePrompt: (transformedText: string) => void
}

export function QuickPrompts({ onUsePrompt }: QuickPromptsProps) {
  const [grouped, setGrouped] = useState(() => listPromptsByCategory())
  const [search, setSearch] = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newPrompt, setNewPrompt] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')
  const [editPrompt, setEditPrompt] = useState('')

  const refresh = useCallback(() => {
    setGrouped(listPromptsByCategory())
  }, [])

  const filtered = useMemo(() => {
    if (!search.trim()) return grouped
    const q = search.toLowerCase()
    const result: Record<string, QuickPrompt[]> = {}
    for (const [cat, prompts] of Object.entries(grouped)) {
      const matched = prompts.filter(p =>
        p.name.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q) ||
        p.prompt.toLowerCase().includes(q)
      )
      if (matched.length > 0) result[cat] = matched
    }
    return result
  }, [grouped, search])

  const handleCreate = () => {
    if (!newName.trim() || !newPrompt.trim()) return
    createPrompt({
      name: newName.trim(),
      description: newDesc.trim() || newName.trim(),
      prompt: newPrompt.trim(),
      icon: '⚡',
      category: 'custom',
    })
    setNewName('')
    setNewDesc('')
    setNewPrompt('')
    setShowCreate(false)
    refresh()
  }

  const handleEdit = (id: string) => {
    if (!editName.trim() || !editPrompt.trim()) return
    updatePrompt(id, {
      name: editName.trim(),
      description: editDesc.trim() || editName.trim(),
      prompt: editPrompt.trim(),
    })
    setEditingId(null)
    refresh()
  }

  const handleDelete = (id: string) => {
    deletePrompt(id)
    refresh()
  }

  const handleUse = (p: QuickPrompt) => {
    onUsePrompt(p.prompt)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Quick Prompts</span>
        <div className="flex gap-1">
          <Button variant="outline" size="sm" className="h-6 text-[10px] px-2" onClick={() => { resetToDefaults(); refresh() }}>
            Reset
          </Button>
          <Button variant="outline" size="sm" className="h-6 text-[10px] px-2" onClick={() => setShowCreate(true)}>
            + New
          </Button>
        </div>
      </div>

      <Input
        placeholder="Search prompts..."
        value={search}
        onChange={e => setSearch(e.target.value)}
        className="h-7 text-xs"
      />

      {showCreate && (
        <div className="space-y-1.5 p-2 rounded border border-border/40 bg-muted/20">
          <Input placeholder="Name" value={newName} onChange={e => setNewName(e.target.value)} className="h-7 text-xs" />
          <Input placeholder="Description (optional)" value={newDesc} onChange={e => setNewDesc(e.target.value)} className="h-7 text-xs" />
          <textarea
            placeholder="Prompt template. Use {{text}} where user input goes."
            value={newPrompt}
            onChange={e => setNewPrompt(e.target.value)}
            className="w-full h-16 resize-none rounded border border-border/60 bg-muted/30 p-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary/40"
          />
          <div className="flex gap-1">
            <Button size="sm" className="h-6 text-[10px] flex-1" onClick={handleCreate}>Save</Button>
            <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={() => setShowCreate(false)}>Cancel</Button>
          </div>
        </div>
      )}

      {Object.keys(filtered).length === 0 ? (
        <p className="text-xs text-muted-foreground text-center py-4">No prompts found</p>
      ) : (
        Object.entries(filtered).map(([category, prompts]) => (
          <div key={category}>
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1 mt-2 first:mt-0">
              {CATEGORY_LABELS[category] || category}
            </div>
            <div className="space-y-1">
              {prompts.map(p => (
                <div key={p.id} className="group rounded border border-border/30 bg-muted/10 p-2 hover:bg-muted/20 transition-colors">
                  {editingId === p.id ? (
                    <div className="space-y-1">
                      <Input value={editName} onChange={e => setEditName(e.target.value)} className="h-7 text-xs" />
                      <Input value={editDesc} onChange={e => setEditDesc(e.target.value)} className="h-7 text-xs" />
                      <textarea value={editPrompt} onChange={e => setEditPrompt(e.target.value)} className="w-full h-16 resize-none rounded border border-border/60 bg-muted/30 p-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary/40" />
                      <div className="flex gap-1">
                        <Button size="sm" className="h-5 text-[10px] px-2 flex-1" onClick={() => handleEdit(p.id)}>Save</Button>
                        <Button variant="outline" size="sm" className="h-5 text-[10px] px-2" onClick={() => setEditingId(null)}>Cancel</Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <button
                        className="w-full text-left"
                        onClick={() => handleUse(p)}
                        title="Click to insert"
                      >
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs">{p.icon}</span>
                          <span className="text-xs font-medium">{p.name}</span>
                          {p.category === 'custom' && (
                            <span className="text-[9px] text-muted-foreground ml-auto">custom</span>
                          )}
                        </div>
                        <div className="text-[10px] text-muted-foreground mt-0.5 line-clamp-1">{p.description}</div>
                      </button>
                      <div className="flex gap-1 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button variant="ghost" size="sm" className="h-5 text-[9px] px-1.5" onClick={() => { setEditingId(p.id); setEditName(p.name); setEditDesc(p.description); setEditPrompt(p.prompt) }}>Edit</Button>
                        {p.category === 'custom' && (
                          <Button variant="ghost" size="sm" className="h-5 text-[9px] px-1.5 text-error hover:text-error" onClick={() => handleDelete(p.id)}>Delete</Button>
                        )}
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  )
}
