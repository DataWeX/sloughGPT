'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'

interface VoicePreset {
  name: string
  rate: number
  pitch: number
  voice: string
}

const STORAGE_KEY = 'sloughgpt-voice-presets'

const DEFAULT_PRESETS: VoicePreset[] = [
  { name: 'Natural', rate: 1.0, pitch: 1.0, voice: '' },
  { name: 'Fast', rate: 1.5, pitch: 1.0, voice: '' },
  { name: 'Slow', rate: 0.7, pitch: 1.0, voice: '' },
  { name: 'Deep', rate: 0.9, pitch: 0.7, voice: '' },
  { name: 'High', rate: 1.0, pitch: 1.4, voice: '' },
]

async function loadPresets(): Promise<VoicePreset[]> {
  try {
    const entry = await chatDB.getKV<VoicePreset[]>(STORAGE_KEY)
    if (entry && Array.isArray(entry) && entry.length > 0) return entry
  } catch { /* corrupted — fall back */ }
  return DEFAULT_PRESETS
}

async function savePresets(presets: VoicePreset[]) {
  try { await chatDB.setKV(STORAGE_KEY, presets) } catch { /* quota exceeded */ }
}

interface VoicePresetCardProps {
  onApply?: (preset: VoicePreset) => void
}

export function VoicePresetCard({ onApply }: VoicePresetCardProps) {
  const [presets, setPresets] = useState<VoicePreset[]>([])
  const [activeName, setActiveName] = useState<string | null>(null)
  const [editing, setEditing] = useState<string | null>(null)
  const [editRate, setEditRate] = useState(1.0)
  const [editPitch, setEditPitch] = useState(1.0)
  const [voices, setVoices] = useState<string[]>([])

  useEffect(() => {
    loadPresets().then(setPresets)
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      const loadVoices = () => {
        const list = window.speechSynthesis.getVoices().map(v => v.name)
        setVoices(list.length > 0 ? list : [])
      }
      loadVoices()
      window.speechSynthesis.onvoiceschanged = loadVoices
    }
  }, [])

  const handleApply = useCallback((preset: VoicePreset) => {
    setActiveName(preset.name)
    onApply?.(preset)
  }, [onApply])

  const handleSaveEdit = () => {
    setPresets(prev => {
      const updated = prev.map(p =>
        p.name === editing ? { ...p, rate: editRate, pitch: editPitch } : p
      )
      savePresets(updated).catch(() => {})
      return updated
    })
    setEditing(null)
  }

  const handleAdd = () => {
    const name = `Preset ${presets.length + 1}`
    const newPreset: VoicePreset = { name, rate: 1.0, pitch: 1.0, voice: '' }
    const updated = [...presets, newPreset]
    setPresets(updated)
    savePresets(updated).catch(() => {})
    setEditing(name)
    setEditRate(1.0)
    setEditPitch(1.0)
  }

  const handleDelete = (name: string) => {
    const updated = presets.filter(p => p.name !== name)
    setPresets(updated)
    savePresets(updated).catch(() => {})
    if (activeName === name) setActiveName(null)
    if (editing === name) setEditing(null)
  }

  const handleTest = (preset: VoicePreset) => {
    if (!('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance('This is a test of the voice preset.')
    utterance.rate = preset.rate
    utterance.pitch = preset.pitch
    if (preset.voice) {
      const match = window.speechSynthesis.getVoices().find(v => v.name === preset.voice)
      if (match) utterance.voice = match
    }
    window.speechSynthesis.speak(utterance)
  }

  if (presets.length === 0) return null

  return (
    <Card data-testid="voice-preset">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Voice Presets</CardTitle>
          <Button size="sm" variant="ghost" onClick={handleAdd}>+ Add</Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-1.5">
          {presets.map(p => (
            <div key={p.name} className="rounded-md border border-border/60 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors">
              {editing === p.name ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-xs">{p.name}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <label className="flex items-center gap-1.5">
                      <span className="text-muted-foreground">Rate</span>
                      <input
                        type="range"
                        min="0.5"
                        max="2.0"
                        step="0.1"
                        value={editRate}
                        onChange={e => setEditRate(parseFloat(e.target.value))}
                        className="w-20"
                      />
                      <span className="font-numeric w-7 text-right">{editRate.toFixed(1)}</span>
                    </label>
                    <label className="flex items-center gap-1.5">
                      <span className="text-muted-foreground">Pitch</span>
                      <input
                        type="range"
                        min="0.5"
                        max="2.0"
                        step="0.1"
                        value={editPitch}
                        onChange={e => setEditPitch(parseFloat(e.target.value))}
                        className="w-20"
                      />
                      <span className="font-numeric w-7 text-right">{editPitch.toFixed(1)}</span>
                    </label>
                  </div>
                  {voices.length > 0 && (
                    <select
                      className="text-xs border border-border rounded px-2 py-1 bg-background w-full"
                      value={p.voice}
                      onChange={e => {
                        const updated = presets.map(pr => pr.name === p.name ? { ...pr, voice: e.target.value } : pr)
                        setPresets(updated)
                        savePresets(updated).catch(() => {})
                      }}
                    >
                      <option value="">Default voice</option>
                      {voices.map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                  )}
                  <div className="flex gap-1">
                    <Button size="sm" variant="ghost" onClick={handleSaveEdit}>Save</Button>
                    <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>Cancel</Button>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button
                      className={`text-left font-medium text-xs px-2 py-0.5 rounded transition-colors ${
                        activeName === p.name
                          ? 'bg-primary/15 text-primary'
                          : 'hover:bg-muted'
                      }`}
                      onClick={() => handleApply(p)}
                    >
                      {p.name}
                    </button>
                    <span className="text-[10px] text-muted-foreground font-numeric">
                      {p.rate.toFixed(1)}x · {p.pitch.toFixed(1)}p
                    </span>
                    {p.voice && (
                      <span className="text-[10px] text-muted-foreground truncate max-w-[100px]">
                        {p.voice}
                      </span>
                    )}
                  </div>
                  <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                    <Button size="sm" variant="ghost" onClick={() => { setEditing(p.name); setEditRate(p.rate); setEditPitch(p.pitch) }}>
                      Edit
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => handleTest(p)}>
                      Test
                    </Button>
                    {!DEFAULT_PRESETS.some(d => d.name === p.name) && (
                      <Button size="sm" variant="ghost" className="text-destructive" onClick={() => handleDelete(p.name)}>
                        Del
                      </Button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
