'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid,
  SearchInput, Slider, Dialog, DialogContent, DialogHeader, DialogTitle,
  Tabs, TabsList, TabsTrigger, TabsContent, AlertDialog, AlertDialogAction,
  AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter,
  AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { IconRefresh, IconPlus, IconTrash } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { soulsController, type Soul, type Checkpoint } from '@/lib/souls-controller'
import { SoulPersonalityCard } from '@/components/souls/SoulPersonalityCard'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'souls' | 'checkpoints' | 'weights' | 'snapshots'

interface ModeInfo {
  label: string
  confidence: number
  scores?: Record<string, number>
  capacity?: number
}

function traitLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function traitColor(value: number): string {
  if (value >= 0.8) return 'text-success'
  if (value >= 0.6) return 'text-primary'
  if (value >= 0.4) return 'text-warning'
  return 'text-muted-foreground'
}

function formatDate(iso: string): string {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

function sourceDir(path: string): string {
  if (!path) return ''
  const parts = path.replace(/\\/g, '/').split('/')
  const modelsIdx = parts.indexOf('models')
  if (modelsIdx >= 0 && modelsIdx + 1 < parts.length) return parts[modelsIdx + 1]
  return parts[parts.length - 2] || ''
}

export default function SoulsPage() {
  const [tab, setTab] = useState<Tab>('souls')
  const [loading, setLoading] = useState(true)

  const [souls, setSouls] = useState<Soul[]>([])
  const [currentSoul, setCurrentSoul] = useState<string | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [modes, setModes] = useState<Record<string, ModeInfo> | null>(null)
  const [snapshots, setSnapshots] = useState<Array<{ name: string; saved_at?: string }>>([])
  const [newSnapshotName, setNewSnapshotName] = useState('')
  const [switching, setSwitching] = useState<string | null>(null)
  const [loadingCheckpoint, setLoadingCheckpoint] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [editedWeights, setEditedWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [savingWeights, setSavingWeights] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [detailSoul, setDetailSoul] = useState<Soul | null>(null)
  const [compareSoul, setCompareSoul] = useState<Soul | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const loadData = useCallback(async () => {
    const [s, c, snaps] = await Promise.all([
      soulsController.list().catch(() => ({ souls: [], current_soul: null })),
      soulsController.listCheckpoints().catch(() => ({ checkpoints: [] })),
      soulsController.listWeightSnapshots().catch(() => []),
    ])
    setSouls(s.souls)
    setCurrentSoul(s.current_soul ?? null)
    setCheckpoints(c.checkpoints)
    setSnapshots(snaps)
  }, [])

  useEffect(() => {
    loadData().finally(() => setLoading(false))
  }, [loadData])

  const handleRefresh = async () => {
    await loadData()
    if (tab === 'weights') await handleLoadWeights()
    if (tab === 'snapshots') await handleLoadSnapshots()
  }

  const handleSwitch = async (name: string) => {
    setSwitching(name)
    try {
      await soulsController.switch(name)
      setCurrentSoul(name)
      addToast(`Switched to ${name}`, 'success')
    } catch {
      addToast('Failed to switch soul', 'error')
    } finally {
      setSwitching(null)
    }
  }

  const handleLoadCheckpoint = async (name: string) => {
    setLoadingCheckpoint(name)
    try {
      await soulsController.loadCheckpoint(name)
      addToast(`Loaded checkpoint: ${name}`, 'success')
      await handleRefresh()
    } catch {
      addToast('Failed to load checkpoint', 'error')
    } finally {
      setLoadingCheckpoint(null)
    }
  }

  const handleDeleteCheckpoint = async (name: string) => {
    try {
      await soulsController.deleteCheckpoint(name)
      addToast(`Deleted checkpoint: ${name}`, 'success')
      setCheckpoints(prev => prev.filter(cp => cp.name !== name))
    } catch {
      addToast('Failed to delete checkpoint', 'error')
    }
    setDeleteTarget(null)
  }

  const handleLoadWeights = async () => {
    try {
      const [w, m] = await Promise.all([
        soulsController.getTraitWeights(),
        soulsController.getModes(),
      ])
      setTraitWeights(w)
      setEditedWeights(JSON.parse(JSON.stringify(w)))
      setModes(m)
    } catch {
      addToast('Failed to load trait weights', 'error')
    }
  }

  const handleWeightChange = (category: string, trait: string, value: number) => {
    if (!editedWeights) return
    const next = { ...editedWeights }
    next[category] = { ...next[category], [trait]: value }
    setEditedWeights(next)
  }

  const handleSaveWeights = async () => {
    if (!editedWeights) return
    setSavingWeights(true)
    try {
      await soulsController.saveTraitWeights(editedWeights)
      setTraitWeights(JSON.parse(JSON.stringify(editedWeights)))
      addToast('Trait weights saved', 'success')
    } catch {
      addToast('Failed to save trait weights', 'error')
    } finally {
      setSavingWeights(false)
    }
  }

  const weightsDirty = JSON.stringify(editedWeights) !== JSON.stringify(traitWeights)

  const handleLoadSnapshots = async () => {
    try {
      setSnapshots(await soulsController.listWeightSnapshots())
    } catch {
      addToast('Failed to load snapshots', 'error')
    }
  }

  const handleSaveSnapshot = async () => {
    if (!newSnapshotName.trim()) return
    try {
      await soulsController.saveWeightSnapshot(newSnapshotName)
      setNewSnapshotName('')
      await handleLoadSnapshots()
      addToast('Snapshot saved', 'success')
    } catch {
      addToast('Failed to save snapshot', 'error')
    }
  }

  const handleLoadSnapshot = async (name: string) => {
    try {
      await soulsController.loadWeightSnapshot(name)
      await handleLoadWeights()
      addToast('Snapshot loaded', 'success')
    } catch {
      addToast('Failed to load snapshot', 'error')
    }
  }

  const handleDeleteSnapshot = async (name: string) => {
    try {
      await soulsController.deleteWeightSnapshot(name)
      await handleLoadSnapshots()
      addToast('Snapshot deleted', 'success')
    } catch {
      addToast('Failed to delete snapshot', 'error')
    }
  }

  const filteredSouls = souls.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.traits?.some(t => t.toLowerCase().includes(searchQuery.toLowerCase())) ||
    s.lineage?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Souls" subtitle="Personality management" />} />
        <div className="space-y-4">
          <KpiGrid>
            <StatCard label="Loading" value="..." />
            <StatCard label="Loading" value="..." />
            <StatCard label="Loading" value="..." />
            <StatCard label="Loading" value="..." />
          </KpiGrid>
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
          <Card><CardContent><div className="h-64 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Souls" subtitle={`${souls.length} personalities · ${currentSoul ?? 'none'} active`} />}
        right={
          <Button size="sm" variant="ghost" onClick={handleRefresh}>
            <IconRefresh className="h-4 w-4 mr-1" /> Refresh
          </Button>
        }
      />

      <KpiGrid>
        <StatCard label="Personalities" value={souls.length} />
        <StatCard label="Active Soul" value={currentSoul ?? 'None'} />
        <StatCard label="Checkpoints" value={checkpoints.length} />
        <StatCard label="Snapshots" value={snapshots.length || '—'} />
      </KpiGrid>

      <Tabs value={tab} onValueChange={(v) => {
        setTab(v as Tab)
        if (v === 'weights') handleLoadWeights()
        if (v === 'snapshots') handleLoadSnapshots()
      }}>
        <TabsList>
          <TabsTrigger value="souls">Souls</TabsTrigger>
          <TabsTrigger value="checkpoints">Checkpoints</TabsTrigger>
          <TabsTrigger value="weights">Weights</TabsTrigger>
          <TabsTrigger value="snapshots">Snapshots</TabsTrigger>
        </TabsList>

        {/* ── Souls Tab ── */}
        <TabsContent value="souls" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Personalities</CardTitle>
              <SearchInput
                value={searchQuery}
                onChange={setSearchQuery}
                placeholder="Search..."
                className="max-w-xs"
              />
            </CardHeader>
            <CardContent>
              {filteredSouls.length === 0 ? (
                <div className="text-center py-8 space-y-2">
                  <p className="text-sm text-muted-foreground">
                    {searchQuery ? 'No personalities match your search.' : 'No personalities found.'}
                  </p>
                  {!searchQuery && (
                    <p className="text-xs text-muted-foreground">
                      Souls are loaded from <code className="bg-muted px-1 rounded">.soul</code> files in <code className="bg-muted px-1 rounded">models/</code>.
                      Train a model or place a soul file to get started.
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  {filteredSouls.map(soul => (
                    <div
                      key={soul.name}
                      className={`flex items-center justify-between rounded-md border px-3 py-2.5 text-sm transition-colors cursor-pointer group ${
                        currentSoul === soul.name
                          ? 'border-primary/40 bg-primary/[0.08]'
                          : 'border-border/60 hover:bg-muted/50'
                      }`}
                      onClick={() => setDetailSoul(soul)}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium truncate">{soul.name}</span>
                          {currentSoul === soul.name && (
                            <span className="text-[10px] font-medium bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">active</span>
                          )}
                          {soul.version && (
                            <span className="text-[10px] font-mono text-muted-foreground">v{soul.version}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                          {soul.lineage && <span>{soul.lineage}</span>}
                          {soul.size_mb != null && soul.size_mb > 0 && <span>{soul.size_mb.toFixed(1)} MB</span>}
                          {soul.born_at && <span>{formatDate(soul.born_at)}</span>}
                          {soul.epochs_trained != null && soul.epochs_trained > 0 && <span>{soul.epochs_trained} epochs</span>}
                          {soul.final_val_loss != null && <span>val {soul.final_val_loss.toFixed(3)}</span>}
                        </div>
                        {soul.traits && soul.traits.length > 0 && (
                          <div className="flex gap-1 mt-1.5 flex-wrap">
                            {soul.traits.slice(0, 5).map(trait => (
                              <span key={trait} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{trait}</span>
                            ))}
                            {soul.traits.length > 5 && (
                              <span className="text-[10px] text-muted-foreground">+{soul.traits.length - 5}</span>
                            )}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1">
                        {soul.personality && Object.keys(soul.personality).length > 0 && (
                          <span className={`text-xs font-mono ${traitColor(Object.values(soul.personality).reduce((a, b) => a + b, 0) / Object.values(soul.personality).length)}`}>
                            {(Object.values(soul.personality).reduce((a, b) => a + b, 0) / Object.values(soul.personality).length * 100).toFixed(0)}%
                          </span>
                        )}
                        {currentSoul !== soul.name && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => { e.stopPropagation(); handleSwitch(soul.name) }}
                            disabled={switching === soul.name}
                          >
                            {switching === soul.name ? 'Switching...' : 'Switch'}
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {(() => {
            const activeSoul = souls.find(s => s.name === currentSoul)
            if (!activeSoul?.personality || Object.keys(activeSoul.personality).length === 0) return null
            return <SoulPersonalityCard personality={activeSoul.personality} traits={activeSoul.traits} soulName={activeSoul.name} />
          })()}
        </TabsContent>

        {/* ── Checkpoints Tab ── */}
        <TabsContent value="checkpoints" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Checkpoints ({checkpoints.length})</CardTitle>
              <Button size="sm" variant="ghost" onClick={handleRefresh}>
                <IconRefresh className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              {checkpoints.length === 0 ? (
                <div className="text-center py-8 space-y-2">
                  <p className="text-sm text-muted-foreground">No checkpoints found.</p>
                  <div className="text-xs text-muted-foreground">
                    <a href="/training" className="text-primary hover:underline">Train a model</a>
                    {' '}to create checkpoints.
                  </div>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {checkpoints.map(cp => (
                    <div key={cp.name} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium truncate">{cp.name}</span>
                          {cp.verdict && (
                            <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                              cp.verdict === 'improved' ? 'bg-success/10 text-success' :
                              cp.verdict === 'degraded' ? 'bg-destructive/10 text-destructive' :
                              'bg-muted text-muted-foreground'
                            }`}>{cp.verdict}</span>
                          )}
                          {cp.is_loaded && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">loaded</span>
                          )}
                          {cp.model_type && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{cp.model_type}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                          {cp.soul && <span>{cp.soul}</span>}
                          {cp.loss != null && <span>loss {cp.loss.toFixed(3)}</span>}
                          {cp.size_mb != null && <span>{cp.size_mb.toFixed(1)} MB</span>}
                          {cp.training_dataset && <span>· {cp.training_dataset.split('/').pop()}</span>}
                          {cp.training_duration_s != null && cp.training_duration_s > 0 && <span>· {cp.training_duration_s.toFixed(0)}s</span>}
                        </div>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                        <Button size="sm" variant="ghost" onClick={() => handleLoadCheckpoint(cp.name)} disabled={loadingCheckpoint === cp.name}>
                          {loadingCheckpoint === cp.name ? (
                            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          ) : (
                            'Load'
                          )}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="text-destructive"
                          onClick={() => setDeleteTarget(cp.name)}
                        >
                          <IconTrash className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Weights Tab ── */}
        <TabsContent value="weights" className="space-y-4">
          {modes && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Context Modes</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {Object.entries(modes).map(([key, mode]) => (
                    <div key={key} className="rounded-md border border-border/60 p-3 space-y-1">
                      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{key}</div>
                      <div className="text-sm font-medium">{mode.label}</div>
                      <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-primary transition-all"
                          style={{ width: `${Math.max(0, Math.min(100, mode.confidence * 100))}%` }}
                        />
                      </div>
                      <div className="text-[10px] text-muted-foreground">
                        {(mode.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div className="flex items-center gap-2">
                <CardTitle className="text-base">Trait Weights</CardTitle>
                {weightsDirty && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-warning/10 text-warning font-medium">unsaved</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {weightsDirty && (
                  <Button size="sm" variant="ghost" onClick={() => setEditedWeights(traitWeights ? JSON.parse(JSON.stringify(traitWeights)) : null)}>
                    Reset
                  </Button>
                )}
                <Button
                  size="sm"
                  onClick={handleSaveWeights}
                  disabled={!weightsDirty || savingWeights}
                >
                  {savingWeights ? 'Saving...' : 'Save'}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {editedWeights ? (
                <div className="space-y-6">
                  {Object.entries(editedWeights).map(([category, weights]) => (
                    <div key={category}>
                      <div className="text-xs font-medium text-muted-foreground mb-3 uppercase tracking-wider">{category}</div>
                      <div className="space-y-3">
                        {Object.entries(weights).map(([trait, value]) => (
                          <Slider
                            key={trait}
                            label={traitLabel(trait)}
                            showValue
                            value={[value]}
                            min={0}
                            max={1}
                            step={0.01}
                            formatValue={(v) => v.toFixed(2)}
                            onValueChange={([v]) => handleWeightChange(category, trait, v)}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Loading trait weights...</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Snapshots Tab ── */}
        <TabsContent value="snapshots" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Weight Snapshots</CardTitle>
              <Button size="sm" variant="ghost" onClick={handleLoadSnapshots}>
                <IconRefresh className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input
                  value={newSnapshotName}
                  onChange={e => setNewSnapshotName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSaveSnapshot()}
                  placeholder="Snapshot name..."
                />
                <Button size="sm" onClick={handleSaveSnapshot} disabled={!newSnapshotName.trim()}>
                  <IconPlus className="h-3.5 w-3.5 mr-1" /> Save
                </Button>
              </div>
              {snapshots.length === 0 ? (
                <div className="text-center py-6">
                  <p className="text-sm text-muted-foreground">No snapshots saved yet.</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Save your current trait weights as a named snapshot to reuse later.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {snapshots.map(snap => (
                    <div key={snap.name} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors">
                      <div className="flex-1 min-w-0">
                        <span className="font-medium truncate">{snap.name}</span>
                        {snap.saved_at && (
                          <span className="text-xs text-muted-foreground ml-2">
                            {new Date(snap.saved_at).toLocaleDateString()} {new Date(snap.saved_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        )}
                      </div>
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                        <Button size="sm" variant="ghost" onClick={() => handleLoadSnapshot(snap.name)}>Load</Button>
                        <Button size="sm" variant="ghost" className="text-destructive" onClick={() => handleDeleteSnapshot(snap.name)}>
                          <IconTrash className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* ── Soul Detail Dialog ── */}
      <Dialog open={detailSoul !== null} onOpenChange={(open) => { if (!open) setDetailSoul(null) }}>
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {detailSoul?.name}
              {detailSoul?.version && <span className="text-xs font-mono text-muted-foreground">v{detailSoul.version}</span>}
            </DialogTitle>
          </DialogHeader>
          {detailSoul && (
            <div className="space-y-4">
              {detailSoul.description && (
                <p className="text-sm text-muted-foreground">{detailSoul.description}</p>
              )}

              {/* Training Metadata */}
              {(detailSoul.born_at || detailSoul.lineage || detailSoul.training_dataset || (detailSoul.epochs_trained != null && detailSoul.epochs_trained > 0) || (detailSoul.size_mb != null && detailSoul.size_mb > 0)) && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Training Info</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {detailSoul.lineage && (
                      <div><span className="text-muted-foreground">Lineage:</span> <span className="font-medium">{detailSoul.lineage}</span></div>
                    )}
                    {detailSoul.base_model && (
                      <div><span className="text-muted-foreground">Base:</span> <span className="font-medium">{detailSoul.base_model}</span></div>
                    )}
                    {detailSoul.born_at && (
                      <div><span className="text-muted-foreground">Created:</span> <span className="font-medium">{formatDate(detailSoul.born_at)}</span></div>
                    )}
                    {(detailSoul.size_mb != null && detailSoul.size_mb > 0) && (
                      <div><span className="text-muted-foreground">Size:</span> <span className="font-medium">{detailSoul.size_mb.toFixed(1)} MB</span></div>
                    )}
                    {(detailSoul.epochs_trained != null && detailSoul.epochs_trained > 0) && (
                      <div><span className="text-muted-foreground">Epochs:</span> <span className="font-medium">{detailSoul.epochs_trained}</span></div>
                    )}
                    {detailSoul.final_train_loss != null && (
                      <div><span className="text-muted-foreground">Train loss:</span> <span className="font-medium">{detailSoul.final_train_loss.toFixed(4)}</span></div>
                    )}
                    {detailSoul.final_val_loss != null && (
                      <div><span className="text-muted-foreground">Val loss:</span> <span className="font-medium">{detailSoul.final_val_loss.toFixed(4)}</span></div>
                    )}
                    {detailSoul.training_dataset && (
                      <div className="col-span-2"><span className="text-muted-foreground">Dataset:</span> <span className="font-medium font-mono text-[10px]">{detailSoul.training_dataset.split('/').pop()}</span></div>
                    )}
                  </div>
                </div>
              )}

              {/* Traits */}
              {detailSoul.traits && detailSoul.traits.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Traits</div>
                  <div className="flex flex-wrap gap-1.5">
                    {detailSoul.traits.map(t => (
                      <span key={t} className="text-xs px-2 py-1 rounded bg-primary/10 text-primary font-medium">{t}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Personality */}
              {detailSoul.personality && Object.keys(detailSoul.personality).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Personality</div>
                  <div className="space-y-2">
                    {Object.entries(detailSoul.personality).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
                      <div key={key}>
                        <div className="flex items-center justify-between text-xs mb-0.5">
                          <span className="text-muted-foreground">{traitLabel(key)}</span>
                          <span className={`font-mono font-medium ${traitColor(value)}`}>{(value * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-primary transition-all"
                            style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Behavior */}
              {detailSoul.behavior && Object.keys(detailSoul.behavior).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Behavior</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(detailSoul.behavior).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-muted-foreground">{traitLabel(key)}: </span>
                        <span className="font-medium">
                          {typeof value === 'number' ? (value * 100).toFixed(0) + '%' : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Cognition */}
              {detailSoul.cognition && Object.keys(detailSoul.cognition).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Cognition</div>
                  <div className="space-y-2">
                    {Object.entries(detailSoul.cognition).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
                      <div key={key}>
                        <div className="flex items-center justify-between text-xs mb-0.5">
                          <span className="text-muted-foreground">{traitLabel(key)}</span>
                          <span className={`font-mono font-medium ${traitColor(value)}`}>{(value * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-1 rounded-full bg-muted/50 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-accent transition-all"
                            style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Emotion */}
              {detailSoul.emotion && Object.keys(detailSoul.emotion).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Emotion</div>
                  <div className="space-y-2">
                    {Object.entries(detailSoul.emotion).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
                      <div key={key}>
                        <div className="flex items-center justify-between text-xs mb-0.5">
                          <span className="text-muted-foreground">{traitLabel(key)}</span>
                          <span className={`font-mono font-medium ${traitColor(value)}`}>{(value * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-1 rounded-full bg-muted/50 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-success transition-all"
                            style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Generation Params */}
              {detailSoul.generation_params && Object.keys(detailSoul.generation_params).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Generation</div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    {Object.entries(detailSoul.generation_params).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-muted-foreground">{traitLabel(key)}: </span>
                        <span className="font-mono font-medium">{typeof value === 'number' ? value : String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Source Path */}
              {detailSoul.path && (
                <div className="text-[10px] text-muted-foreground font-mono truncate pt-1 border-t border-border/30">
                  {detailSoul.path}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <Button size="sm" variant="outline" onClick={() => setDetailSoul(null)}>Close</Button>
                {currentSoul !== detailSoul.name && (
                  <Button size="sm" onClick={() => { handleSwitch(detailSoul.name); setDetailSoul(null) }}>
                    Switch to {detailSoul.name}
                  </Button>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ── Delete Checkpoint Confirmation ── */}
      <AlertDialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open) setDeleteTarget(null) }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete checkpoint?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{deleteTarget}</strong>. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => deleteTarget && handleDeleteCheckpoint(deleteTarget)}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
