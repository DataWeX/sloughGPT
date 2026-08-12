'use client'

import { useState } from 'react'
import { Button } from '@sloughgpt/strui'
import { BUILT_IN_PRESETS, type TrainingPreset } from '@/hooks/useTrainingForm'

export function TrainingPresets({
  onApply,
  customPresets,
  onSave,
  onDelete,
}: {
  onApply: (preset: TrainingPreset) => void
  customPresets: TrainingPreset[]
  onSave: (preset: TrainingPreset) => void
  onDelete: (name: string) => void
}) {
  const [saving, setSaving] = useState(false)
  const [presetName, setPresetName] = useState('')

  const allPresets = [...BUILT_IN_PRESETS, ...customPresets]

  const handleSave = () => {
    const name = presetName.trim()
    if (!name) return
    onSave({ name, description: 'Custom preset', method: 'distill', epochs: 5, lr: 1e-3, batchSize: 32 })
    setPresetName('')
    setSaving(false)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Presets</span>
        <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setSaving(!saving)}>
          {saving ? 'Cancel' : 'Save current'}
        </Button>
      </div>

      {saving && (
        <div className="flex items-center gap-1.5">
          <input
            value={presetName}
            onChange={e => setPresetName(e.target.value)}
            placeholder="Preset name..."
            className="h-7 flex-1 rounded-md border border-border/60 bg-background px-2 text-xs"
            onKeyDown={e => e.key === 'Enter' && handleSave()}
          />
          <Button size="sm" className="h-7 text-xs" onClick={handleSave} disabled={!presetName.trim()}>Save</Button>
        </div>
      )}

      <div className="flex flex-wrap gap-1.5">
        {allPresets.map(p => {
          const isCustom = customPresets.some(cp => cp.name === p.name)
          return (
            <button
              key={p.name}
              onClick={() => onApply(p)}
              className="group relative flex items-center gap-1 rounded-md border border-border/40 bg-muted/30 px-2 py-1 text-[11px] hover:bg-primary/10 hover:border-primary/30 transition-colors"
            >
              <span className="font-medium">{p.name}</span>
              <span className="text-muted-foreground/60 hidden sm:inline">{p.description}</span>
              {isCustom && (
                <span
                  role="button"
                  tabIndex={0}
                  className="ml-0.5 text-destructive/50 hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={e => { e.stopPropagation(); onDelete(p.name) }}
                  onKeyDown={e => { if (e.key === 'Enter') { e.stopPropagation(); onDelete(p.name) } }}
                >
                  x
                </span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
