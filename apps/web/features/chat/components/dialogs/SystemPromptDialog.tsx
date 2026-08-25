'use client'

import { useState, useEffect, useCallback } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconTrash, IconPlus } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'

interface SystemPromptDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  value: string
  onSave: (value: string) => void
}

interface Preset {
  name: string
  prompt: string
}

const STORAGE_KEY = 'chat:system_prompt_presets'

async function loadPresets(): Promise<Preset[]> {
  const stored = await chatDB.getKV<Preset[]>(STORAGE_KEY)
  return stored ?? []
}

async function savePresets(presets: Preset[]) {
  await chatDB.setKV(STORAGE_KEY, presets)
}

const DEFAULTS: Preset[] = [
  { name: 'Helpful Assistant', prompt: 'You are a helpful, harmless, and honest assistant. Answer concisely and accurately.' },
  { name: 'Code Expert', prompt: 'You are an expert software engineer. Provide clean, well-documented code solutions. Explain your reasoning.' },
  { name: 'Creative Writer', prompt: 'You are a creative writing partner. Use vivid language and imaginative descriptions. Be expressive and engaging.' },
  { name: 'Tutor', prompt: 'You are a patient tutor. Explain concepts step by step. Use analogies and examples. Ask questions to check understanding.' },
]

export function SystemPromptDialog({ open, onOpenChange, value, onSave }: SystemPromptDialogProps) {
  const [draft, setDraft] = useState(value)
  const [presets, setPresets] = useState<Preset[]>([])
  const [presetName, setPresetName] = useState('')
  const [showSaveInput, setShowSaveInput] = useState(false)

  useEffect(() => {
    let active = true
    loadPresets().then(stored => {
      if (active) {
        if (stored.length === 0) {
          savePresets(DEFAULTS)
          setPresets(DEFAULTS)
        } else {
          setPresets(stored)
        }
      }
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    setDraft(value)
  }, [value, open])

  const handleOpenChange = (next: boolean) => {
    if (!next) setDraft(value)
    onOpenChange(next)
  }

  const handleSave = () => {
    onSave(draft)
    onOpenChange(false)
  }

  const applyPreset = useCallback((p: Preset) => {
    setDraft(p.prompt)
  }, [])

  const handleSaveAsPreset = async () => {
    const name = presetName.trim()
    if (!name || !draft.trim()) return
    const updated = [...presets.filter(p => p.name !== name), { name, prompt: draft }]
    setPresets(updated)
    await savePresets(updated)
    setPresetName('')
    setShowSaveInput(false)
  }

  const handleDeletePreset = async (name: string) => {
    const updated = presets.filter(p => p.name !== name)
    setPresets(updated)
    await savePresets(updated)
  }

  const hasUnsavedDraft = draft !== value

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Custom System Prompt</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Override the system prompt for this conversation. Saved presets appear below.
            </p>

            {presets.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Presets</p>
                <div className="flex flex-wrap gap-1.5">
                  {presets.map(p => (
                    <div key={p.name} className="group flex items-center gap-1 rounded-md border border-border/50 bg-muted/20 px-2 py-1">
                      <button
                        type="button"
                        onClick={() => applyPreset(p)}
                        className="text-[11px] text-foreground/80 hover:text-foreground whitespace-nowrap"
                        title={p.prompt}
                      >
                        {p.name}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDeletePreset(p.name)}
                        className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 text-muted-foreground hover:text-destructive transition-all"
                        title="Delete preset"
                        aria-label="Delete preset"
                      >
                        <IconTrash className="h-2.5 w-2.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <textarea
              className="w-full min-h-[120px] text-sm p-2 rounded-md border border-border bg-background resize-y focus:outline-none focus:ring-2 focus:ring-primary/30"
              placeholder="Enter instructions for the AI..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              aria-label="System prompt"
            />

            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5">
                {showSaveInput ? (
                  <div className="flex items-center gap-1">
                    <Input
                      aria-label="Preset name"
                      value={presetName}
                      onChange={e => setPresetName(e.target.value)}
                      placeholder="Preset name..."
                      className="h-7 text-[11px] w-32"
                      onKeyDown={e => { if (e.key === 'Enter') handleSaveAsPreset() }}
                    />
                    <Button size="sm" className="h-7 text-[10px] px-2" onClick={handleSaveAsPreset} disabled={!presetName.trim() || !draft.trim()}>
                      Save
                    </Button>
                    <Button size="sm" variant="ghost" className="h-7 text-[10px] px-1" onClick={() => { setShowSaveInput(false); setPresetName('') }}>
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <Button size="sm" variant="ghost" className="h-7 text-[11px]" onClick={() => setShowSaveInput(true)} disabled={!draft.trim()}>
                    <IconPlus className="h-3 w-3 mr-1" /> Save as preset
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={() => handleOpenChange(false)}>Cancel</Button>
                <Button size="sm" onClick={handleSave} disabled={!hasUnsavedDraft && draft === value}>
                  {hasUnsavedDraft ? 'Save changes' : 'Saved'}
                </Button>
              </div>
            </div>
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  )
}
