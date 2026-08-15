'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, SearchInput } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { soulsController, type Soul, type Checkpoint } from '@/lib/souls-controller'
import { SoulPersonalityCard } from '@/components/souls/SoulPersonalityCard'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'souls' | 'checkpoints' | 'weights' | 'snapshots'

export default function SoulsPage() {
  const [tab, setTab] = useState<Tab>('souls')
  const [loading, setLoading] = useState(true)

  const [souls, setSouls] = useState<Soul[]>([])
  const [currentSoul, setCurrentSoul] = useState<string | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [snapshots, setSnapshots] = useState<Array<{ name: string; saved_at?: string }>>([])
  const [newSnapshotName, setNewSnapshotName] = useState('')
  const [switching, setSwitching] = useState<string | null>(null)
  const [loadingCheckpoint, setLoadingCheckpoint] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    Promise.all([
      soulsController.list().catch(() => ({ souls: [], current_soul: null })),
      soulsController.listCheckpoints().catch(() => ({ checkpoints: [] })),
      soulsController.listWeightSnapshots().catch(() => []),
    ]).then(([s, c, snaps]) => {
      setSouls(s.souls)
      setCurrentSoul(s.current_soul ?? null)
      setCheckpoints(c.checkpoints)
      setSnapshots(snaps)
    }).finally(() => setLoading(false))
  }, [])

  const handleRefresh = async () => {
    const [s, c] = await Promise.all([
      soulsController.list().catch(() => ({ souls: [], current_soul: null })),
      soulsController.listCheckpoints().catch(() => ({ checkpoints: [] })),
    ])
    setSouls(s.souls)
    setCurrentSoul(s.current_soul ?? null)
    setCheckpoints(c.checkpoints)
  }

  const handleSwitch = async (name: string) => {
    setSwitching(name)
    try {
      await soulsController.switch(name)
      setCurrentSoul(name)
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
      await handleRefresh()
    } catch {
      addToast('Failed to load checkpoint', 'error')
    } finally {
      setLoadingCheckpoint(null)
    }
  }

  const handleLoadWeights = async () => {
    try {
      const weights = await soulsController.getTraitWeights()
      setTraitWeights(weights)
    } catch {
      addToast('Failed to load trait weights', 'error')
    }
  }

  const handleLoadSnapshots = async () => {
    try {
      const snaps = await soulsController.listWeightSnapshots()
      setSnapshots(snaps)
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
    } catch {
      addToast('Failed to save snapshot', 'error')
    }
  }

  const handleLoadSnapshot = async (name: string) => {
    try {
      await soulsController.loadWeightSnapshot(name)
      await handleLoadWeights()
    } catch {
      addToast('Failed to load snapshot', 'error')
    }
  }

  const handleDeleteSnapshot = async (name: string) => {
    try {
      await soulsController.deleteWeightSnapshot(name)
      await handleLoadSnapshots()
    } catch {
      addToast('Failed to delete snapshot', 'error')
    }
  }

  if (loading) {
    return (
      <PageContainer
        title="Souls"
        subtitle="Personality management"
        loadingContent={
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
        }
      >
        <></>
      </PageContainer>
    )
  }

  const filteredSouls = souls.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.traits?.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const toolbar = (
    <SearchInput
      value={searchQuery}
      onChange={setSearchQuery}
      placeholder="Search personalities..."
      className="max-w-sm"
    />
  )

  return (
    <PageContainer
      title="Souls"
      subtitle={`${souls.length} personalities · ${currentSoul ?? 'none'} active`}
      toolbar={toolbar}
    >
        <KpiGrid>
          <StatCard label="Personalities" value={souls.length} />
          <StatCard label="Active Soul" value={currentSoul ?? 'None'} />
          <StatCard label="Checkpoints" value={checkpoints.length} />
          <StatCard label="Snapshots" value={snapshots.length || '—'} />
        </KpiGrid>

        <div className="flex gap-1 border-b border-border/30 pb-0">
          {(['souls', 'checkpoints', 'weights', 'snapshots'] as Tab[]).map(t => (
            <button
              key={t}
              onClick={() => {
                setTab(t)
                if (t === 'weights') handleLoadWeights()
                if (t === 'snapshots') handleLoadSnapshots()
              }}
              className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors ${
                tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        {tab === 'souls' && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Personalities</CardTitle>
              <Button size="sm" variant="ghost" onClick={handleRefresh}>
                <IconRefresh className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              {filteredSouls.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  {searchQuery ? 'No personalities match your search.' : 'No personalities found.'}
                </p>
              ) : (
                <div className="space-y-2">
                  {filteredSouls.map(soul => (
                    <div
                      key={soul.name}
                      className={`flex items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors ${
                        currentSoul === soul.name
                          ? 'border-primary/40 bg-primary/[0.08]'
                          : 'border-border/60 hover:bg-muted/50'
                      }`}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium truncate">{soul.name}</span>
                          {currentSoul === soul.name && (
                            <span className="text-xs bg-primary/10 text-primary px-1 rounded">active</span>
                          )}
                        </div>
                        {soul.description && (
                          <div className="text-xs text-muted-foreground mt-0.5 truncate">{soul.description}</div>
                        )}
                        {soul.traits && soul.traits.length > 0 && (
                          <div className="flex gap-1 mt-1 flex-wrap">
                            {soul.traits.slice(0, 4).map(trait => (
                              <span key={trait} className="text-xs bg-muted px-1 rounded">{trait}</span>
                            ))}
                          </div>
                        )}
                      </div>
                      {currentSoul !== soul.name && (
                        <Button size="sm" variant="ghost" onClick={() => handleSwitch(soul.name)} disabled={switching === soul.name}>
                          {switching === soul.name ? 'Switching...' : 'Switch'}
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {tab === 'souls' && (() => {
          const activeSoul = souls.find(s => s.name === currentSoul)
          if (!activeSoul?.personality || Object.keys(activeSoul.personality).length === 0) return null
          return <SoulPersonalityCard personality={activeSoul.personality} traits={activeSoul.traits} soulName={activeSoul.name} />
        })()}

        {tab === 'checkpoints' && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Checkpoints ({checkpoints.length})</CardTitle>
              <Button size="sm" variant="ghost" onClick={handleRefresh}>
                <IconRefresh className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              {checkpoints.length === 0 ? (
                <div className="text-center py-6 space-y-2">
                  <p className="text-sm text-muted-foreground">No checkpoints found.</p>
                  <a href="/training" className="text-sm text-primary hover:underline">Train a model</a>
                  <span className="text-sm text-muted-foreground"> to create checkpoints.</span>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {checkpoints.map(cp => (
                    <div key={cp.name} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium truncate">{cp.name}</span>
                          {cp.verdict && (
                            <span className={`text-xs px-1 rounded ${
                              cp.verdict === 'improved' ? 'bg-success/10 text-success' :
                              cp.verdict === 'degraded' ? 'bg-destructive/10 text-destructive' :
                              'bg-muted text-muted-foreground'
                            }`}>{cp.verdict}</span>
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {cp.soul && <span>{cp.soul} · </span>}
                          {cp.loss != null && <span>loss {cp.loss.toFixed(3)} · </span>}
                          {cp.size_mb != null && <span>{cp.size_mb.toFixed(1)} MB</span>}
                        </div>
                      </div>
                      <Button size="sm" variant="ghost" onClick={() => handleLoadCheckpoint(cp.name)} disabled={loadingCheckpoint === cp.name}>
                        {loadingCheckpoint === cp.name ? 'Loading...' : 'Load'}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {tab === 'weights' && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Trait Weights</CardTitle>
              <Button size="sm" variant="ghost" onClick={handleLoadWeights}>
                <IconRefresh className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              {traitWeights ? (
                <div className="space-y-4">
                  {Object.entries(traitWeights).map(([category, weights]) => (
                    <div key={category}>
                      <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">{category}</div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                        {Object.entries(weights).map(([trait, value]) => (
                          <div key={trait} className="rounded-md bg-muted/30 px-3 py-2">
                            <div className="text-xs text-muted-foreground capitalize">{trait.replace(/_/g, ' ')}</div>
                            <div className="text-sm font-mono font-medium">{typeof value === 'number' ? value.toFixed(2) : String(value)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Click refresh to load trait weights.</p>
              )}
            </CardContent>
          </Card>
        )}

        {tab === 'snapshots' && (
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
                <Button size="sm" onClick={handleSaveSnapshot} disabled={!newSnapshotName.trim()}>Save</Button>
              </div>
              {snapshots.length === 0 ? (
                <p className="text-sm text-muted-foreground">No snapshots saved yet.</p>
              ) : (
                <div className="space-y-2">
                  {snapshots.map(snap => (
                    <div key={snap.name} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors">
                      <div className="flex-1 min-w-0">
                        <span className="font-medium truncate">{snap.name}</span>
                        {snap.saved_at && (
                          <span className="text-xs text-muted-foreground ml-2">{new Date(snap.saved_at).toLocaleDateString()}</span>
                        )}
                      </div>
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                        <Button size="sm" variant="ghost" onClick={() => handleLoadSnapshot(snap.name)}>Load</Button>
                        <Button size="sm" variant="ghost" className="text-destructive" onClick={() => handleDeleteSnapshot(snap.name)}>Delete</Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
    </PageContainer>
  )
}
