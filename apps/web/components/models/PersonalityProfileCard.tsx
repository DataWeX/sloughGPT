'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { IconPlus } from '@sloughgpt/strui'
import SoulVisualizer from '@/components/souls/SoulVisualizer'
import TraitEditor from '@/components/souls/TraitEditor'
import { soulsController } from '@/lib/souls-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

interface SnapshotMeta {
  name: string; saved_at?: string; label?: string
}

interface PersonalityProfileCardProps {
  traitWeights: Record<string, Record<string, number>> | null
  currentSoulName?: string | null
  onTraitsSaved: (weights: Record<string, Record<string, number>>) => Promise<void>
  onTraitsChanged: () => Promise<void>
}

export default function PersonalityProfileCard({
  traitWeights, currentSoulName, onTraitsSaved, onTraitsChanged,
}: PersonalityProfileCardProps) {
  const addToast = useToastStore(s => s.addToast)
  const [editingTraits, setEditingTraits] = useState(false)
  const [snapshots, setSnapshots] = useState<SnapshotMeta[]>([])
  const [snapshotName, setSnapshotName] = useState('')

  const fetchSnapshots = async () => {
    try {
      const list = await soulsController.listWeightSnapshots()
      setSnapshots(list)
    } catch { addToast('Could not load weight snapshots', 'info') }
  }

  const handleSaveSnapshot = async () => {
    const name = snapshotName.trim()
    if (!name) return
    try {
      await soulsController.saveWeightSnapshot(name)
      setSnapshotName('')
      addToast(`Saved "${name}"`, 'success')
      await fetchSnapshots()
    } catch (err) {
      addToast(extractErrorMessage(err, 'Failed to save'), 'error')
    }
  }

  const handleLoadSnapshot = async (name: string) => {
    try {
      const count = await soulsController.loadWeightSnapshot(name)
      addToast(`Loaded "${name}" (${count} traits)`, 'success')
      await onTraitsChanged()
    } catch (err) {
      addToast(extractErrorMessage(err, 'Failed to load'), 'error')
    }
  }

  const handleDeleteSnapshot = async (name: string) => {
    if (!confirm(`Delete snapshot "${name}"? This cannot be undone.`)) return
    try {
      await soulsController.deleteWeightSnapshot(name)
      addToast(`Deleted "${name}"`, 'success')
      await fetchSnapshots()
    } catch (err) {
      addToast(extractErrorMessage(err, 'Failed to delete'), 'error')
    }
  }

  if (!traitWeights) return null

  return (
    <Card className="relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/[0.02] to-transparent pointer-events-none" />
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Personality Profile</CardTitle>
          <button
            type="button"
            onClick={() => setEditingTraits(!editingTraits)}
            className="text-[10px] px-2 py-1 rounded-md border border-border/60 hover:bg-muted/50 transition-colors"
          >
            {editingTraits ? 'View' : 'Edit'}
          </button>
        </div>
      </CardHeader>
      <CardContent className="relative">
        <p className="text-[10px] text-muted-foreground mb-3">Traits shape how your personality responds &mdash; like a character sheet for your AI.</p>
        {editingTraits ? (
          <TraitEditor
            traitWeights={traitWeights}
            onSave={onTraitsSaved}
            onReset={() => setEditingTraits(false)}
          />
        ) : (
          <SoulVisualizer traitWeights={traitWeights} currentSoulName={currentSoulName ?? null} />
        )}

        <div className="mt-4 pt-3 border-t border-border/40">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              Snapshots ({snapshots.length})
            </span>
          </div>
          <div className="flex items-center gap-2 mb-3">
            <Input
              value={snapshotName}
              onChange={e => setSnapshotName(e.target.value)}
              placeholder="Name this state..."
              className="h-7 text-[11px]"
              onKeyDown={e => { if (e.key === 'Enter') handleSaveSnapshot() }}
            />
            <Button size="sm" className="h-7 text-[11px] px-2 shrink-0" onClick={handleSaveSnapshot} disabled={!snapshotName.trim()}>
              <IconPlus className="w-3 h-3 mr-1" /> Save
            </Button>
          </div>
          {snapshots.length === 0 ? (
            <div className="text-[10px] text-muted-foreground">Save weight presets to switch between personalities quickly</div>
          ) : (
            <div className="space-y-1">
              {snapshots.map(s => (
                <div key={s.name} className="flex items-center justify-between px-2 py-1.5 rounded bg-muted/30 hover:bg-primary/[0.05] transition-colors group">
                  <div className="min-w-0 flex-1">
                    <span className="text-[11px] font-medium">{s.label || s.name}</span>
                    {s.saved_at && (
                      <span className="text-[9px] text-muted-foreground ml-2">
                        {new Date(s.saved_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button type="button" className="text-[10px] text-primary hover:text-primary/80 px-1.5 py-0.5 rounded hover:bg-primary/10" onClick={() => handleLoadSnapshot(s.name)}>Load</button>
                    <button type="button" className="text-[10px] text-destructive hover:text-destructive/80 px-1.5 py-0.5 rounded hover:bg-destructive/10" onClick={() => handleDeleteSnapshot(s.name)}>Delete</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
