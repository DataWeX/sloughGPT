'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { IconRefresh, IconPlus, IconTrash } from '@/components/ui'
import { Input } from '@/components/ui/input'
import { KpiGrid, StatCard, Chip, Badge, InlineBanner, SectionHeader } from '@/components/strui'
import { useApiHealth } from '@/hooks/useApiHealth'
import { catalogIdMatchesRuntime } from '@/lib/inference-display'
import { useToastStore } from '@/lib/toast-store'
import { modelController } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import { generateController } from '@/lib/generate-controller'
import type { Soul, Checkpoint } from '@/lib/souls-controller'
import { cn } from '@/lib/cn'
import SoulVisualizer from '@/components/souls/SoulVisualizer'
import TraitEditor from '@/components/souls/TraitEditor'
import {
  useModels,
  useLoadModel,
  useSouls,
  useCheckpoints,
  useCurrentSoul,
  useSwitchSoul,
} from '@/lib/query/api-hooks'

interface Model {
  id: string
  name: string
  source?: string
  description?: string
  tags?: string[]
  size_mb?: number
  size_gb?: number
  params?: string
  cached?: boolean
  thumbnail?: string
}

const FORMAT_BADGES: Record<string, { label: string; variant?: string }> = {
  pytorch: { label: 'PyTorch' },
  gguf: { label: 'GGUF' },
  onnx: { label: 'ONNX' },
  sou: { label: '.sou' },
}

export default function ModelsPage() {
  const router = useRouter()
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [warmingModel, setWarmingModel] = useState<string | null>(null)
  const [switchingSoul, setSwitchingSoul] = useState<string | null>(null)
  const [traitWeights, setTraitWeights] = useState<Record<string, any> | null>(null)
  const [mounted, setMounted] = useState(false)
  const [modelSearch, setModelSearch] = useState('')
  const { state: health, refresh: refreshHealth } = useApiHealth()
  const addToast = useToastStore(s => s.addToast)

  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useModels()
  const { data: soulsData, isLoading: soulsLoading, refetch: refetchSouls } = useSouls()
  const { data: currentSoulData, isLoading: currentSoulLoading, refetch: refetchCurrentSoul } = useCurrentSoul()
  const { data: checkpointsData, isLoading: checkpointsLoading, refetch: refetchCheckpoints } = useCheckpoints()
  const { mutateAsync: loadModel } = useLoadModel()
  const { mutateAsync: switchSoul } = useSwitchSoul()

  const models = (modelsData ?? []) as Model[]
  const souls = soulsData?.souls ?? []
  const currentSoul = currentSoulData?.name ?? soulsData?.current_soul ?? null
  const checkpoints = checkpointsData?.checkpoints ?? []
  const activeCheckpoint = checkpointsData?.active_checkpoint ?? null
  const activeRuntimeId = health !== null && health !== 'offline' && health.model_loaded ? health.model_type : null

  const handleLoadModel = async (modelId: string) => {
    setLoadingModel(modelId)
    try {
      const data = await loadModel(modelId)
      if (data.error) {
        addToast(data.error, 'error')
        return
      }
      addToast(`${data.model_id ?? modelId} ready`, 'success')
      setLoadingModel(null)
      setWarmingModel(modelId)
      try {
        await generateController.generate({ prompt: 'Hello', max_new_tokens: 2 })
      } catch {
        // warmup failure is non-fatal
      }
      setWarmingModel(null)
    } catch (err) {
      addToast(err instanceof Error ? err.message : String(err), 'error')
      setLoadingModel(null)
    } finally {
      setLoadingModel(null)
      setWarmingModel(null)
      await refreshHealth()
      await refetchModels()
    }
  }

  const handleSwitchSoul = async (name: string, checkpointName?: string) => {
    setSwitchingSoul(name)
    try {
      await switchSoul({ name, checkpointName })
      addToast(checkpointName ? `${name} + ${checkpointName}` : name, 'success')
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed', 'error')
    } finally {
      setSwitchingSoul(null)
    }
  }

  const handleSaveTraits = useCallback(async (weights: Record<string, Record<string, number>>) => {
    try {
      await soulsController.saveTraitWeights(weights)
      addToast('Personality updated', 'success')
      setTraitWeights(weights)
      setEditingTraits(false)
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to save traits', 'error')
    }
  }, [addToast])

  const handleRefresh = async () => {
    setRefreshing(true)
    await Promise.allSettled([
      refetchModels(),
      refetchSouls(),
      refetchCurrentSoul(),
      refetchCheckpoints(),
      refreshHealth(),
      fetchTraitWeights(),
      fetchSnapshots(),
    ])
    setRefreshing(false)
    addToast('Refreshed', 'success')
  }

  const fetchTraitWeights = useCallback(async () => {
    try {
      const w = await soulsController.getTraitWeights()
      if (w && !('error' in w)) setTraitWeights(w)
    } catch { addToast('Could not load trait weights', 'info') }
  }, [addToast])

  // ── Snapshots ──
  interface SnapshotMeta {
    name: string
    saved_at?: string
    label?: string
  }
  const [editingTraits, setEditingTraits] = useState(false)
  const [snapshots, setSnapshots] = useState<SnapshotMeta[]>([])
  const [snapshotName, setSnapshotName] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [cacheUsage, setCacheUsage] = useState<{ total_gb: number; model_count: number } | null>(null)

  const fetchSnapshots = useCallback(async () => {
    try {
      const list = await soulsController.listWeightSnapshots()
      setSnapshots(list)
    } catch { addToast('Could not load weight snapshots', 'info') }
  }, [addToast])

  const handleSaveSnapshot = async () => {
    const name = snapshotName.trim()
    if (!name) return
    try {
      await soulsController.saveWeightSnapshot(name)
      setSnapshotName('')
      addToast(`Saved "${name}"`, 'success')
      await fetchSnapshots()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to save', 'error')
    }
  }

  const handleLoadSnapshot = async (name: string) => {
    try {
      const count = await soulsController.loadWeightSnapshot(name)
      addToast(`Loaded "${name}" (${count} traits)`, 'success')
      await fetchTraitWeights()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to load', 'error')
    }
  }

  const handleDeleteSnapshot = async (name: string) => {
    if (!confirm(`Delete snapshot "${name}"? This cannot be undone.`)) return
    try {
      await soulsController.deleteWeightSnapshot(name)
      addToast(`Deleted "${name}"`, 'success')
      await fetchSnapshots()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to delete', 'error')
    }
  }

  useEffect(() => { setMounted(true); fetchTraitWeights(); fetchSnapshots(); modelController.getCacheUsage().then(setCacheUsage).catch(() => {}) }, [fetchTraitWeights, fetchSnapshots])

  const isOnline = health !== null && health !== 'offline'
  const subtitle = health === null ? 'Connecting...'
    : !isOnline ? 'API offline'
    : health.model_loaded ? `${health.model_type} · running`
    : 'No model loaded'

  const pipelineSteps = []
  if (activeRuntimeId) pipelineSteps.push({ label: 'Base Model', value: activeRuntimeId })
  if (currentSoul) pipelineSteps.push({ label: 'Personality', value: currentSoul })
  if (activeCheckpoint) pipelineSteps.push({ label: 'Checkpoint', value: activeCheckpoint })

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <style>{`
        @keyframes pulseGlow {
          0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.3); }
          50% { box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.1); }
        }
        @keyframes cardLift {
          to { transform: translateY(-2px); box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        }
        .card-hover {
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .card-hover:hover {
          animation: cardLift 0.2s ease forwards;
        }
        .pulse-dot {
          animation: pulseGlow 2s ease-in-out infinite;
        }
      `}</style>
      <AppRouteHeader
        className="items-start"
        left={<AppRouteHeaderLead title="Models & Personalities" subtitle={subtitle} />}
        right={<Button type="button" variant="secondary" size="sm" disabled={refreshing} onClick={handleRefresh}><IconRefresh className={`w-3.5 h-3.5 mr-1 ${refreshing ? 'animate-spin' : ''}`} /> {refreshing ? 'Refreshing...' : 'Refresh'}</Button>}
      />

      <div className="space-y-4">

        {/* ── Active Pipeline ── */}
        {isOnline && (
          <Card>
            <CardHeader><CardTitle className="text-base">Active Pipeline</CardTitle></CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className={cn("w-2 h-2 rounded-full", health.model_loaded ? "bg-success pulse-dot" : "bg-warning")} />
                  <span className="text-xs font-medium">{health.model_type || 'No model'}</span>
                </div>
                {currentSoul && (
                  <>
                    <span className="text-muted-foreground/40 text-xs">→</span>
                    <Chip variant="primary">{currentSoul}</Chip>
                  </>
                )}
                {activeCheckpoint && (
                  <>
                    <span className="text-muted-foreground/40 text-xs">→</span>
                    <Chip>{activeCheckpoint}</Chip>
                  </>
                )}
                {health.inference_count != null && (
                  <span className="ml-auto text-[10px] text-muted-foreground">{health.inference_count} inferences</span>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Pipeline Stats ── */}
        <KpiGrid columns={3}>
          <StatCard label="Models" value={modelsLoading ? '—' : models.length.toString()} icon={<span className="text-xs font-mono">M</span>} />
          <StatCard label="Personalities" value={soulsLoading ? '—' : souls.length.toString()} icon={<span className="text-xs">🎭</span>} />
          <StatCard label="Checkpoints" value={checkpointsLoading ? '—' : checkpoints.length.toString()} icon={<span className="text-xs">📦</span>} />
        </KpiGrid>

        {/* ── Composable Layers ── */}
        <Card>
          <CardHeader><CardTitle className="text-base">Composable Layers</CardTitle></CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {[
                { title: 'Base Models', desc: 'Load any HuggingFace model as the foundation layer', icon: '🧠', count: models.length },
                { title: 'Personalities', desc: 'Soul profiles that wrap the model with traits & voice', icon: '🎭', count: souls.length },
                { title: 'Adapters', desc: 'LoRA/DoRA fine-tuned adapters that stack on any base', icon: '🧩', count: checkpoints.filter((c: Checkpoint) => c.soul).length },
                { title: 'Checkpoints', desc: 'Trained checkpoints that persist a model+personality snapshot', icon: '📦', count: checkpoints.length },
              ].map(layer => (
                <div key={layer.title} className="rounded-lg border border-border/60 p-3 card-hover">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-base">{layer.icon}</span>
                    <span className="text-sm font-medium">{layer.title}</span>
                  </div>
                  <p className="text-[11px] text-muted-foreground mb-2">{layer.desc}</p>
                  <span className="text-[10px] text-muted-foreground/60">{layer.count} available</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* ── Personalities ── */}
        {(souls.length > 0 || soulsLoading) && (
          <Card>
            <CardHeader><CardTitle className="text-base">Personalities</CardTitle></CardHeader>
            <CardContent className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {soulsLoading ? (
                [1,2,3].map(i => (
                  <div key={i} className="animate-pulse flex items-center gap-3 p-3 rounded-lg border border-border/60">
                    <div className="w-2 h-2 rounded-full bg-muted-foreground/20" />
                    <div className="flex-1 space-y-1.5">
                      <div className="h-3 w-24 bg-muted rounded" />
                      <div className="h-2.5 w-32 bg-muted rounded" />
                    </div>
                    <div className="h-6 w-14 bg-muted rounded" />
                  </div>
                ))
              ) : (
              souls.map((s: Soul) => {
                const soulCheckpoints = checkpoints?.filter((c: Checkpoint) => c.soul === s.name) ?? []
                const isCurrent = currentSoul === s.name
                return (
                  <div key={s.name} className={cn("flex items-center justify-between p-3 rounded-lg border card-hover", isCurrent ? "border-primary/40 bg-primary/5" : "border-border/60")}>
                    <div className="flex items-center gap-3 min-w-0">
                      <div className={cn("w-2 h-2 rounded-full shrink-0", isCurrent ? "bg-primary" : "bg-muted-foreground/30")} />
                      <div className="min-w-0">
                        <div className="text-sm font-medium truncate">{s.name}</div>
                        <div className="text-xs text-muted-foreground truncate">{s.description || ''}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {soulCheckpoints.length > 0 ? (
                        <select
                          className="h-7 text-xs rounded-md border border-border/60 bg-transparent px-2"
                          value={isCurrent && activeCheckpoint ? activeCheckpoint : ''}
                          onChange={(e) => {
                            const val = e.target.value
                            if (val === '__base__') handleSwitchSoul(s.name)
                            else if (val) handleSwitchSoul(s.name, val)
                          }}
                          disabled={switchingSoul === s.name}
                        >
                          <option value="__base__">Switch</option>
                          {soulCheckpoints.map((cp: Checkpoint) => (
                            <option key={cp.name} value={cp.name}>{cp.name}{activeCheckpoint === cp.name ? ' (active)' : ''}</option>
                          ))}
                        </select>
                      ) : checkpointsLoading ? (
                        <span className="text-xs text-muted-foreground animate-pulse">loading&hellip;</span>
                      ) : isCurrent ? (
                        <Badge variant="success">Active</Badge>
                      ) : (
                        <Button variant="outline" size="sm" className="h-7 text-xs px-2" disabled={switchingSoul === s.name} onClick={() => handleSwitchSoul(s.name)}>
                          {switchingSoul === s.name ? '...' : 'Switch'}
                        </Button>
                      )}
                    </div>
                  </div>
                )
              })
              )}
            </CardContent>
          </Card>
        )}

        {/* ── Personality Profile ── */}
        {traitWeights && mounted && (
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
                  onSave={handleSaveTraits}
                  onReset={() => setEditingTraits(false)}
                />
              ) : (
                <SoulVisualizer traitWeights={traitWeights} currentSoulName={currentSoul} />
              )}

              {/* ── Snapshots section ── */}
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
                      <div key={s.name} className="flex items-center justify-between px-2 py-1.5 rounded bg-muted/30 hover:bg-muted/50 transition-colors">
                        <div className="min-w-0 flex-1">
                          <span className="text-[11px] font-medium">{s.label || s.name}</span>
                          {s.saved_at && (
                            <span className="text-[9px] text-muted-foreground ml-2">
                              {new Date(s.saved_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <button
                            type="button"
                            className="text-[10px] text-primary hover:text-primary/80 px-1.5 py-0.5 rounded hover:bg-primary/10"
                            onClick={() => handleLoadSnapshot(s.name)}
                            title="Load this snapshot"
                          >Load</button>
                          <button
                            type="button"
                            className="text-[10px] text-red-400 hover:text-red-300 px-1.5 py-0.5 rounded hover:bg-red-500/10"
                            onClick={() => handleDeleteSnapshot(s.name)}
                            title="Delete snapshot"
                          >Delete</button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Model Catalog ── */}
        <Card>
          <CardHeader><CardTitle className="text-base">Model Catalog</CardTitle></CardHeader>
          <CardContent>
            {modelsLoading ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[1,2,3,4,5,6].map(i => (
                  <div key={i} className="animate-pulse rounded-lg bg-muted/50 border border-border/50 p-4 h-20" />
                ))}
              </div>
            ) : models.length === 0 ? (
              <div className="text-center py-8 text-sm text-muted-foreground">No models available</div>
            ) : (() => {
              const filtered = modelSearch
                ? models.filter(m => (m.name || m.id).toLowerCase().includes(modelSearch.toLowerCase()) || m.id.toLowerCase().includes(modelSearch.toLowerCase()))
                : models
              return (
                <>
                  {models.length > 6 && (
                    <div className="mb-3">
                      <input
                        type="text"
                        value={modelSearch}
                        onChange={e => setModelSearch(e.target.value)}
                        placeholder="Search models..."
                        className="h-8 w-full max-w-xs rounded-md border border-border/60 bg-background px-2 text-xs placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
                      />
                    </div>
                  )}
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {filtered.map((model) => {
                  const isLoading = loadingModel === model.id
                  const isLoaded = activeRuntimeId ? catalogIdMatchesRuntime(model.id, activeRuntimeId) : false
                  return (
                    <div
                      key={model.id}
                      className={cn(
                        "flex items-center justify-between p-3 rounded-lg border cursor-pointer card-hover",
                        isLoaded ? "border-primary/40 bg-primary/5" : "border-border/60"
                      )}
                      onClick={() => router.push(`/model/${encodeURIComponent(model.id)}`)}
                      onKeyDown={(e) => { if (e.key === 'Enter') router.push(`/model/${encodeURIComponent(model.id)}`) }}
                      tabIndex={0}
                      role="button"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="text-sm font-medium truncate">{model.name || model.id}</div>
                        <div className="flex items-center gap-2 mt-0.5">
                          {model.params && <span className="text-[10px] text-muted-foreground font-mono">{model.params}</span>}
                          {model.size_gb && <span className="text-[10px] text-muted-foreground">{(model.size_gb).toFixed(1)} GB</span>}
                          {isLoaded && <span className="text-[10px] text-primary font-medium">Loaded</span>}
                          {!isLoaded && model.cached && <span className="text-[10px] text-success/70">Cached</span>}
                        </div>
                      </div>
                      <div className="flex gap-1 shrink-0 ml-2">
                        {model.source === 'local' ? (
                          <span className="text-[10px] text-muted-foreground">Local</span>
                        ) : (
                          <Button size="sm" variant={isLoaded ? 'outline' : 'default'} className="h-7 text-xs px-3" disabled={isLoading || warmingModel === model.id} onClick={(e) => { e.stopPropagation(); handleLoadModel(model.id) }}>
                            {isLoading ? '…' : warmingModel === model.id ? 'Warming…' : isLoaded ? 'Loaded' : 'Load'}
                          </Button>
                        )}
                      </div>
                    </div>
                  )
                })}
                  </div>
                  {filtered.length === 0 && modelSearch && (
                    <div className="text-center py-4 text-xs text-muted-foreground">No models matching &quot;{modelSearch}&quot;</div>
                  )}
                </>
              )
            })()}
          </CardContent>
        </Card>

        {/* Cache Usage */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Model Cache</CardTitle>
            <Button size="sm" variant="ghost" onClick={() => modelController.getCacheUsage().then(setCacheUsage).catch(() => {})}>
              <IconRefresh className="h-3.5 w-3.5" />
            </Button>
          </CardHeader>
          <CardContent>
            {cacheUsage ? (
              <KpiGrid columns={3}>
                <StatCard label="Cached Models" value={cacheUsage.model_count} />
                <StatCard label="Disk Usage" value={`${cacheUsage.total_gb.toFixed(1)} GB`} />
                <StatCard label="Loaded" value={health !== null && health !== 'offline' && health.model_loaded ? health.model_type || 'Yes' : 'None'} />
              </KpiGrid>
            ) : (
              <p className="text-sm text-muted-foreground">Loading cache stats...</p>
            )}
          </CardContent>
        </Card>
      </div>


    </div>
  )
}
