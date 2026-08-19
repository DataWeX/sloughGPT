'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid,
  SearchInput, Slider, Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription,
  Tabs, TabsList, TabsTrigger, TabsContent, AlertDialog, AlertDialogAction,
  AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter,
  AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { IconRefresh, IconPlus, IconTrash, IconDownload } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { soulsController, type Soul, type Checkpoint } from '@/lib/souls-controller'
import { SoulPersonalityCard } from '@/components/souls/SoulPersonalityCard'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'souls' | 'checkpoints' | 'weights' | 'snapshots' | 'analytics'

interface ModeInfo {
  label: string
  confidence: number
  scores?: Record<string, number>
  capacity?: number
}

interface SoulStats {
  total_souls: number
  current_soul: string | null
  available_souls: string[]
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

function formatDateTime(iso: string): string {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
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

function verdictBadge(verdict: string): { className: string; label: string } {
  switch (verdict) {
    case 'improved': return { className: 'bg-success/10 text-success', label: 'Improved' }
    case 'degraded': return { className: 'bg-destructive/10 text-destructive', label: 'Degraded' }
    case 'neutral': return { className: 'bg-muted text-muted-foreground', label: 'Neutral' }
    default: return { className: 'bg-muted text-muted-foreground', label: verdict }
  }
}

// ── Trait Radar (mini visualization) ──

function TraitRadar({ values, size = 80 }: { values: Record<string, number>; size?: number }) {
  const entries = Object.entries(values).slice(0, 8)
  if (entries.length === 0) return null
  const cx = size / 2
  const cy = size / 2
  const r = size / 2 - 4
  const angleStep = (2 * Math.PI) / entries.length

  const points = entries.map(([_, v], i) => {
    const angle = i * angleStep - Math.PI / 2
    const dist = r * Math.max(0.1, Math.min(1, v))
    return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle) }
  })

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z'

  return (
    <svg width={size} height={size} className="shrink-0">
      {[0.25, 0.5, 0.75, 1].map(scale => (
        <polygon
          key={scale}
          points={entries.map((_, i) => {
            const angle = i * angleStep - Math.PI / 2
            return `${cx + r * scale * Math.cos(angle)},${cy + r * scale * Math.sin(angle)}`
          }).join(' ')}
          fill="none"
          stroke="currentColor"
          className="text-border/40"
          strokeWidth={0.5}
        />
      ))}
      <polygon points={pathD} fill="rgb(var(--primary))" fillOpacity={0.15} stroke="rgb(var(--primary))" strokeWidth={1} />
    </svg>
  )
}

// ── Personality Bar ──

function PersonalityBar({ label, value, color = 'bg-primary' }: { label: string; value: number; color?: string }) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-0.5">
        <span className="text-muted-foreground">{label}</span>
        <span className={`font-mono font-medium ${traitColor(value)}`}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
      </div>
    </div>
  )
}

// ── Main Page ──

export default function SoulsPage() {
  const [tab, setTab] = useState<Tab>('souls')
  const [loading, setLoading] = useState(true)

  // ── Data ──
  const [souls, setSouls] = useState<Soul[]>([])
  const [currentSoul, setCurrentSoul] = useState<string | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [modes, setModes] = useState<Record<string, ModeInfo> | null>(null)
  const [snapshots, setSnapshots] = useState<Array<{ name: string; saved_at?: string }>>([])
  const [stats, setStats] = useState<SoulStats | null>(null)

  // ── UI State ──
  const [searchQuery, setSearchQuery] = useState('')
  const [switching, setSwitching] = useState<string | null>(null)
  const [loadingCheckpoint, setLoadingCheckpoint] = useState<string | null>(null)
  const [newSnapshotName, setNewSnapshotName] = useState('')
  const [editedWeights, setEditedWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [savingWeights, setSavingWeights] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [deleteType, setDeleteType] = useState<'checkpoint' | 'snapshot'>('checkpoint')

  // ── Dialogs ──
  const [detailSoul, setDetailSoul] = useState<Soul | null>(null)
  const [compareSoul, setCompareSoul] = useState<Soul | null>(null)
  const [checkpointDetail, setCheckpointDetail] = useState<Checkpoint | null>(null)

  const addToast = useToastStore(s => s.addToast)

  // ── Data Loading ──
  const loadData = useCallback(async () => {
    const [s, c, snaps, st] = await Promise.all([
      soulsController.list().catch(() => ({ souls: [], current_soul: null })),
      soulsController.listCheckpoints().catch(() => ({ checkpoints: [] })),
      soulsController.listWeightSnapshots().catch(() => []),
      soulsController.getStats().catch(() => null),
    ])
    setSouls(s.souls)
    setCurrentSoul(s.current_soul ?? null)
    setCheckpoints(c.checkpoints)
    setSnapshots(snaps)
    if (st) setStats(st)
  }, [])

  useEffect(() => {
    loadData().finally(() => setLoading(false))
  }, [loadData])

  const handleRefresh = async () => {
    await loadData()
    if (tab === 'weights') await handleLoadWeights()
    if (tab === 'snapshots') await handleLoadSnapshots()
  }

  // ── Soul Actions ──
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

  // ── Checkpoint Actions ──
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

  const handleDownloadCheckpoint = async (name: string) => {
    try {
      const blob = await soulsController.downloadCheckpoint(name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${name}.soul`
      a.click()
      URL.revokeObjectURL(url)
      addToast(`Downloaded ${name}`, 'success')
    } catch {
      addToast('Download failed', 'error')
    }
  }

  const handleCheckpointInfo = async (name: string) => {
    try {
      const info = await soulsController.checkpointInfo(name)
      if (info) setCheckpointDetail(info)
    } catch {
      addToast('Failed to load checkpoint info', 'error')
    }
  }

  // ── Weight Actions ──
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

  // ── Snapshot Actions ──
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
    setDeleteTarget(null)
  }

  // ── Filtered Data ──
  const filteredSouls = useMemo(() => souls.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.traits?.some(t => t.toLowerCase().includes(searchQuery.toLowerCase())) ||
    s.lineage?.toLowerCase().includes(searchQuery.toLowerCase())
  ), [souls, searchQuery])

  const filteredCheckpoints = useMemo(() => checkpoints.filter(cp =>
    cp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    cp.soul?.toLowerCase().includes(searchQuery.toLowerCase())
  ), [checkpoints, searchQuery])

  // ── Loading State ──
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
        </div>
      </div>
    )
  }

  // ── Render ──
  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Souls" subtitle={`${souls.length} personalities · ${currentSoul ?? 'none'} active`} />}
        right={
          <Button size="sm" variant="ghost" onClick={handleRefresh}>
            <IconRefresh className="h-4 w-4" />
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
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        {/* ═══════════════ SOULS TAB ═══════════════ */}
        <TabsContent value="souls" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Personalities</CardTitle>
              <SearchInput
                value={searchQuery}
                onChange={setSearchQuery}
                placeholder="Search souls..."
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
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <TraitRadar values={soul.personality || {}} size={48} />
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
                            {soul.training_dataset && (
                              <span className="font-mono text-[10px]">{sourceDir(soul.training_dataset)}</span>
                            )}
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

        {/* ═══════════════ CHECKPOINTS TAB ═══════════════ */}
        <TabsContent value="checkpoints" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Checkpoints ({checkpoints.length})</CardTitle>
              <div className="flex gap-2">
                <SearchInput
                  value={searchQuery}
                  onChange={setSearchQuery}
                  placeholder="Search checkpoints..."
                  className="max-w-xs"
                />
                <Button size="sm" variant="ghost" onClick={handleRefresh}>
                  <IconRefresh className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {filteredCheckpoints.length === 0 ? (
                <div className="text-center py-8 space-y-2">
                  <p className="text-sm text-muted-foreground">No checkpoints found.</p>
                  <div className="text-xs text-muted-foreground">
                    <a href="/training" className="text-primary hover:underline">Train a model</a>
                    {' '}to create checkpoints.
                  </div>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {filteredCheckpoints.map(cp => (
                    <div key={cp.name} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2.5 text-sm group hover:bg-muted/50 transition-colors">
                      <div className="flex-1 min-w-0 cursor-pointer" onClick={() => handleCheckpointInfo(cp.name)}>
                        <div className="flex items-center gap-2">
                          <span className="font-medium truncate">{cp.name}</span>
                          {cp.verdict && (
                            <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${verdictBadge(cp.verdict).className}`}>
                              {verdictBadge(cp.verdict).label}
                            </span>
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
                          {cp.born_at && <span>· {formatDate(cp.born_at)}</span>}
                        </div>
                        {cp.perplexity_delta != null && cp.perplexity_delta !== 0 && (
                          <div className="flex items-center gap-3 text-xs mt-0.5">
                            <span className={cp.perplexity_delta < 0 ? 'text-success' : 'text-destructive'}>
                              PPL {cp.perplexity_delta > 0 ? '+' : ''}{cp.perplexity_delta.toFixed(3)}
                            </span>
                            {cp.bleu_delta != null && cp.bleu_delta !== 0 && (
                              <span className={cp.bleu_delta > 0 ? 'text-success' : 'text-destructive'}>
                                BLEU {cp.bleu_delta > 0 ? '+' : ''}{cp.bleu_delta.toFixed(3)}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                        <Button size="sm" variant="ghost" onClick={() => handleLoadCheckpoint(cp.name)} disabled={loadingCheckpoint === cp.name}>
                          {loadingCheckpoint === cp.name ? (
                            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          ) : 'Load'}
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => handleDownloadCheckpoint(cp.name)}>
                          <IconDownload className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="text-destructive"
                          onClick={() => { setDeleteTarget(cp.name); setDeleteType('checkpoint') }}
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

        {/* ═══════════════ WEIGHTS TAB ═══════════════ */}
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
                <Button size="sm" onClick={handleSaveWeights} disabled={!weightsDirty || savingWeights}>
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

        {/* ═══════════════ SNAPSHOTS TAB ═══════════════ */}
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
                <div className="text-center py-6 space-y-1">
                  <p className="text-sm text-muted-foreground">No snapshots saved yet.</p>
                  <p className="text-xs text-muted-foreground">
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
                            {formatDateTime(snap.saved_at)}
                          </span>
                        )}
                      </div>
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                        <Button size="sm" variant="ghost" onClick={() => handleLoadSnapshot(snap.name)}>Load</Button>
                        <Button size="sm" variant="ghost" className="text-destructive" onClick={() => { setDeleteTarget(snap.name); setDeleteType('snapshot') }}>
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

        {/* ═══════════════ ANALYTICS TAB ═══════════════ */}
        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Soul Overview</CardTitle>
            </CardHeader>
            <CardContent>
               {souls.length === 0 ? (
                 <div className="text-center py-8 space-y-2">
                   <p className="text-sm text-muted-foreground">No souls available for analysis.</p>
                   <p className="text-xs text-muted-foreground">
                     Register a soul from a <code className="bg-muted px-1 rounded">.soul</code> file to see analytics here.
                   </p>
                 </div>
               ) : (
                <div className="space-y-4">
                  {/* Per-soul summary */}
                  <div className="space-y-2">
                    {souls.map(soul => {
                      const avgPersonality = soul.personality && Object.keys(soul.personality).length > 0
                        ? Object.values(soul.personality).reduce((a, b) => a + b, 0) / Object.values(soul.personality).length
                        : 0
                      return (
                        <div key={soul.name} className="flex items-center gap-3 rounded-md border border-border/60 px-3 py-2 text-sm">
                          <TraitRadar values={soul.personality || {}} size={40} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{soul.name}</span>
                              {currentSoul === soul.name && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">active</span>
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {soul.lineage || 'Unknown lineage'} · {soul.size_mb != null ? `${soul.size_mb.toFixed(1)} MB` : '—'}
                              {soul.born_at ? ` · ${formatDate(soul.born_at)}` : ''}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className={`text-sm font-mono font-medium ${traitColor(avgPersonality)}`}>
                              {(avgPersonality * 100).toFixed(0)}%
                            </div>
                            <div className="text-[10px] text-muted-foreground">avg personality</div>
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {/* Personality comparison chart */}
                  {souls.length > 1 && (
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Personality Comparison</div>
                      <div className="grid grid-cols-1 gap-1">
                        {(() => {
                          const allTraits = new Set<string>()
                          souls.forEach(s => s.personality && Object.keys(s.personality).forEach(t => allTraits.add(t)))
                          const traits = Array.from(allTraits).slice(0, 10)
                          return traits.map(trait => (
                            <div key={trait} className="flex items-center gap-2 text-xs">
                              <span className="w-24 text-muted-foreground truncate">{traitLabel(trait)}</span>
                              <div className="flex-1 flex gap-1">
                                {souls.map(soul => {
                                  const val = soul.personality?.[trait] ?? 0
                                  return (
                                    <div key={soul.name} className="flex-1">
                                      <div className="h-2 rounded-full bg-muted/50 overflow-hidden">
                                        <div
                                          className="h-full rounded-full bg-primary/60 transition-all"
                                          style={{ width: `${Math.max(0, Math.min(100, val * 100))}%` }}
                                        />
                                      </div>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          ))
                        })()}
                      </div>
                      <div className="flex gap-3 mt-2 text-[10px] text-muted-foreground">
                        {souls.map(soul => (
                          <span key={soul.name} className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-primary/60" />
                            {soul.name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Checkpoint summary */}
                  {checkpoints.length > 0 && (
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Checkpoint Summary</div>
                      <div className="grid grid-cols-3 gap-3 text-sm">
                        <div className="rounded-md border border-border/60 p-2 text-center">
                          <div className="text-lg font-bold">{checkpoints.length}</div>
                          <div className="text-[10px] text-muted-foreground">Total</div>
                        </div>
                        <div className="rounded-md border border-border/60 p-2 text-center">
                          <div className="text-lg font-bold text-success">
                            {checkpoints.filter(c => c.verdict === 'improved').length}
                          </div>
                          <div className="text-[10px] text-muted-foreground">Improved</div>
                        </div>
                        <div className="rounded-md border border-border/60 p-2 text-center">
                          <div className="text-lg font-bold text-destructive">
                            {checkpoints.filter(c => c.verdict === 'degraded').length}
                          </div>
                          <div className="text-[10px] text-muted-foreground">Degraded</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* ═══════════════ DIALOGS ═══════════════ */}

      {/* Soul Detail Dialog */}
      <Dialog open={detailSoul !== null} onOpenChange={(open) => { if (!open) setDetailSoul(null) }}>
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {detailSoul?.name}
              {detailSoul?.version && <span className="text-xs font-mono text-muted-foreground">v{detailSoul.version}</span>}
            </DialogTitle>
            {detailSoul?.description && (
              <DialogDescription>{detailSoul.description}</DialogDescription>
            )}
          </DialogHeader>
          {detailSoul && (
            <div className="space-y-4">
              {/* Training Metadata */}
              {(detailSoul.born_at || detailSoul.lineage || detailSoul.training_dataset || (detailSoul.epochs_trained != null && detailSoul.epochs_trained > 0) || (detailSoul.size_mb != null && detailSoul.size_mb > 0)) && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Training Info</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {detailSoul.lineage && <div><span className="text-muted-foreground">Lineage:</span> <span className="font-medium">{detailSoul.lineage}</span></div>}
                    {detailSoul.base_model && <div><span className="text-muted-foreground">Base:</span> <span className="font-medium">{detailSoul.base_model}</span></div>}
                    {detailSoul.born_at && <div><span className="text-muted-foreground">Created:</span> <span className="font-medium">{formatDate(detailSoul.born_at)}</span></div>}
                    {(detailSoul.size_mb != null && detailSoul.size_mb > 0) && <div><span className="text-muted-foreground">Size:</span> <span className="font-medium">{detailSoul.size_mb.toFixed(1)} MB</span></div>}
                    {(detailSoul.epochs_trained != null && detailSoul.epochs_trained > 0) && <div><span className="text-muted-foreground">Epochs:</span> <span className="font-medium">{detailSoul.epochs_trained}</span></div>}
                    {detailSoul.final_train_loss != null && <div><span className="text-muted-foreground">Train loss:</span> <span className="font-medium">{detailSoul.final_train_loss.toFixed(4)}</span></div>}
                    {detailSoul.final_val_loss != null && <div><span className="text-muted-foreground">Val loss:</span> <span className="font-medium">{detailSoul.final_val_loss.toFixed(4)}</span></div>}
                    {detailSoul.training_dataset && <div className="col-span-2"><span className="text-muted-foreground">Dataset:</span> <span className="font-medium font-mono text-[10px]">{detailSoul.training_dataset.split('/').pop()}</span></div>}
                  </div>
                </div>
              )}

              {/* Personality */}
              {detailSoul.personality && Object.keys(detailSoul.personality).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Personality</div>
                  <div className="space-y-2">
                    {Object.entries(detailSoul.personality).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
                      <PersonalityBar key={key} label={traitLabel(key)} value={value} />
                    ))}
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
                      <PersonalityBar key={key} label={traitLabel(key)} value={value} color="bg-accent" />
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
                      <PersonalityBar key={key} label={traitLabel(key)} value={value} color="bg-success" />
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

      {/* Compare Dialog */}
      <Dialog open={compareSoul !== null} onOpenChange={(open) => { if (!open) setCompareSoul(null) }}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Compare: {currentSoul} vs {compareSoul?.name}</DialogTitle>
          </DialogHeader>
          {compareSoul && (() => {
            const activeSoul = souls.find(s => s.name === currentSoul)
            if (!activeSoul) return null
            const allTraits = new Set<string>()
            Object.keys(activeSoul.personality || {}).forEach(t => allTraits.add(t))
            Object.keys(compareSoul.personality || {}).forEach(t => allTraits.add(t))
            const traits = Array.from(allTraits)
            return (
              <div className="space-y-3">
                {traits.map(trait => {
                  const v1 = activeSoul.personality?.[trait] ?? 0
                  const v2 = compareSoul.personality?.[trait] ?? 0
                  const diff = v2 - v1
                  return (
                    <div key={trait} className="flex items-center gap-3 text-xs">
                      <span className="w-28 text-muted-foreground truncate">{traitLabel(trait)}</span>
                      <div className="flex-1 flex items-center gap-2">
                        <span className="w-10 text-right font-mono">{(v1 * 100).toFixed(0)}%</span>
                        <div className="flex-1 h-2 rounded-full bg-muted/50 overflow-hidden relative">
                          <div className="absolute inset-0 h-full rounded-full bg-primary/40" style={{ width: `${v1 * 100}%` }} />
                          <div className="absolute inset-0 h-full rounded-full bg-accent/60" style={{ width: `${v2 * 100}%` }} />
                        </div>
                        <span className="w-10 font-mono">{(v2 * 100).toFixed(0)}%</span>
                      </div>
                      <span className={`w-12 text-right font-mono ${diff > 0 ? 'text-success' : diff < 0 ? 'text-destructive' : 'text-muted-foreground'}`}>
                        {diff > 0 ? '+' : ''}{(diff * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
                <div className="flex gap-4 text-[10px] text-muted-foreground pt-2 border-t border-border/30">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-primary/40" /> {activeSoul.name}</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-accent/60" /> {compareSoul.name}</span>
                </div>
              </div>
            )
          })()}
        </DialogContent>
      </Dialog>

      {/* Checkpoint Detail Dialog */}
      <Dialog open={checkpointDetail !== null} onOpenChange={(open) => { if (!open) setCheckpointDetail(null) }}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {checkpointDetail?.name}
              {checkpointDetail?.verdict && (
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${verdictBadge(checkpointDetail.verdict).className}`}>
                  {verdictBadge(checkpointDetail.verdict).label}
                </span>
              )}
            </DialogTitle>
          </DialogHeader>
          {checkpointDetail && (
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-2">
                {checkpointDetail.soul && <div><span className="text-muted-foreground">Soul:</span> <span className="font-medium">{checkpointDetail.soul}</span></div>}
                {checkpointDetail.model_type && <div><span className="text-muted-foreground">Type:</span> <span className="font-medium">{checkpointDetail.model_type}</span></div>}
                {checkpointDetail.loss != null && <div><span className="text-muted-foreground">Loss:</span> <span className="font-mono">{checkpointDetail.loss.toFixed(4)}</span></div>}
                {checkpointDetail.size_mb != null && <div><span className="text-muted-foreground">Size:</span> <span className="font-mono">{checkpointDetail.size_mb.toFixed(1)} MB</span></div>}
                {checkpointDetail.epochs != null && <div><span className="text-muted-foreground">Epochs:</span> <span className="font-mono">{checkpointDetail.epochs}</span></div>}
                {checkpointDetail.steps != null && <div><span className="text-muted-foreground">Steps:</span> <span className="font-mono">{checkpointDetail.steps}</span></div>}
                {checkpointDetail.training_dataset && <div className="col-span-2"><span className="text-muted-foreground">Dataset:</span> <span className="font-mono">{checkpointDetail.training_dataset}</span></div>}
                {checkpointDetail.born_at && <div className="col-span-2"><span className="text-muted-foreground">Created:</span> <span className="font-medium">{formatDateTime(checkpointDetail.born_at)}</span></div>}
              </div>
              {checkpointDetail.perplexity_delta != null && checkpointDetail.perplexity_delta !== 0 && (
                <div className="flex gap-4 pt-1 border-t border-border/30">
                  <span className={checkpointDetail.perplexity_delta < 0 ? 'text-success' : 'text-destructive'}>
                    Perplexity: {checkpointDetail.perplexity_delta > 0 ? '+' : ''}{checkpointDetail.perplexity_delta.toFixed(3)}
                  </span>
                  {checkpointDetail.bleu_delta != null && checkpointDetail.bleu_delta !== 0 && (
                    <span className={checkpointDetail.bleu_delta > 0 ? 'text-success' : 'text-destructive'}>
                      BLEU: {checkpointDetail.bleu_delta > 0 ? '+' : ''}{checkpointDetail.bleu_delta.toFixed(3)}
                    </span>
                  )}
                </div>
              )}
              <div className="flex justify-end gap-2 pt-2">
                <Button size="sm" variant="outline" onClick={() => setCheckpointDetail(null)}>Close</Button>
                <Button size="sm" variant="ghost" onClick={() => { if (checkpointDetail) handleDownloadCheckpoint(checkpointDetail.name) }}>
                  <IconDownload className="h-3.5 w-3.5 mr-1" /> Download
                </Button>
                {currentSoul !== checkpointDetail.soul && (
                  <Button size="sm" onClick={() => { if (checkpointDetail) { handleLoadCheckpoint(checkpointDetail.name); setCheckpointDetail(null) } }}>
                    Load
                  </Button>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open) setDeleteTarget(null) }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {deleteType}?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{deleteTarget}</strong>. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                if (!deleteTarget) return
                if (deleteType === 'checkpoint') handleDeleteCheckpoint(deleteTarget)
                else handleDeleteSnapshot(deleteTarget)
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
