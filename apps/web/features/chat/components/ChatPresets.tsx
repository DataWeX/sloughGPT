'use client'

import { useState, useCallback, useMemo, useEffect, memo } from 'react'
import { Button, IconX, IconPlus, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface Preset {
  id: string
  name: string
  prompt: string
  createdAt: number
}

interface ChatPresetsProps {
  onSelect: (prompt: string) => void
  className?: string
}

const STORAGE_KEY = 'chat-presets'

function loadPresets(): Preset[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function savePresets(presets: Preset[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets))
}

export const ChatPresets = memo(function ChatPresets({
  onSelect,
  className,
}: ChatPresetsProps) {
  const [presets, setPresets] = useState<Preset[]>([])
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [prompt, setPrompt] = useState('')
  const [justCreated, setJustCreated] = useState<string | null>(null)

  useEffect(() => {
    setPresets(loadPresets())
  }, [])

  const handleCreate = useCallback(() => {
    const trimmedName = name.trim()
    const trimmedPrompt = prompt.trim()
    if (!trimmedName || !trimmedPrompt) return

    const newPreset: Preset = {
      id: crypto.randomUUID(),
      name: trimmedName,
      prompt: trimmedPrompt,
      createdAt: Date.now(),
    }

    const next = [...presets, newPreset]
    setPresets(next)
    savePresets(next)
    setName('')
    setPrompt('')
    setCreating(false)
    setJustCreated(newPreset.id)
    setTimeout(() => setJustCreated(null), 1500)
  }, [name, prompt, presets])

  const handleDelete = useCallback((id: string) => {
    const next = presets.filter(p => p.id !== id)
    setPresets(next)
    savePresets(next)
  }, [presets])

  const handleSelect = useCallback((presetPrompt: string) => {
    onSelect(presetPrompt)
  }, [onSelect])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Prompt Presets</span>
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
        <div className="p-2 border-b space-y-2">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Preset name..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Prompt template..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 resize-none focus:outline-none focus:ring-1 focus:ring-primary/50 min-h-[60px]"
          />
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-6"
              onClick={handleCreate}
              disabled={!name.trim() || !prompt.trim()}
            >
              <IconCheck className="h-3 w-3 mr-1" />
              Save
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

      <div className="max-h-[300px] overflow-y-auto">
        {presets.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No presets yet. Click + to create one.
          </p>
        ) : (
          <div className="divide-y">
            {presets.map(preset => (
              <div
                key={preset.id}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 hover:bg-muted/30 group',
                  justCreated === preset.id && 'bg-success/10',
                )}
              >
                <button
                  type="button"
                  className="flex-1 text-left min-w-0"
                  onClick={() => handleSelect(preset.prompt)}
                >
                  <div className="text-xs font-medium truncate">{preset.name}</div>
                  <div className="text-[10px] text-muted-foreground truncate">
                    {preset.prompt.slice(0, 80)}{preset.prompt.length > 80 ? '…' : ''}
                  </div>
                </button>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="h-5 w-5 opacity-0 group-hover:opacity-100 shrink-0"
                  onClick={() => handleDelete(preset.id)}
                  title="Delete preset"
                >
                  <IconX className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})