'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, StatCard, KpiGrid, SearchInput, Skeleton, cn } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { registryController, type RegisteredModel, type RegistryStats } from '@/lib/registry-controller'
import { RegistryHealthCard } from '@/components/registry/RegistryHealthCard'
import { useToastStore } from '@/lib/toast-store'

export default function RegistryPage() {
  const [models, setModels] = useState<RegisteredModel[]>([])
  const [stats, setStats] = useState<RegistryStats | null>(null)
  const [bestModel, setBestModel] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedModel, setExpandedModel] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchData = useCallback(async () => {
    try {
      const [modelsRes, statsRes, bestRes] = await Promise.all([
        registryController.list().catch(() => []),
        registryController.stats().catch(() => null),
        registryController.best().catch(() => null),
      ])
      setModels(modelsRes)
      setStats(statsRes)
      setBestModel(bestRes)
    } catch {
      addToast('Could not load registry data', 'error')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])

  const filteredModels = models.filter(m =>
    m.model_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.status.toLowerCase().includes(searchQuery.toLowerCase())
  )

  if (loading) {
    return (
      <PageContainer
        title="Registry"
        subtitle="Model registry"
        loadingContent={
          <div className="space-y-4">
            <KpiGrid>
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
            </KpiGrid>
            <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
            <Card><CardContent><div className="h-48 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
          </div>
        }
      ><></>
      </PageContainer>
    )
  }

  const toolbar = (
    <SearchInput
      value={searchQuery}
      onChange={setSearchQuery}
      placeholder="Search models..."
      className="max-w-sm"
    />
  )

  return (
    <PageContainer
      title="Registry"
      subtitle={`${models.length} models registered`}
      toolbar={toolbar}
    >
        <KpiGrid>
          <StatCard label="Total Models" value={stats?.total_models ?? 0} />
          <StatCard label="Loaded" value={stats?.loaded_models ?? 0} />
          <StatCard label="Failed" value={stats?.failed_models ?? 0} />
          <StatCard label="Circuit Breaker" value={stats?.circuit_breaker_open ? 'Open' : 'Closed'} />
        </KpiGrid>

        <RegistryHealthCard models={models} stats={stats} />

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Registered Models</CardTitle>
            <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={fetchData} aria-label="Refresh">
              <IconRefresh className="h-3 w-3" />
            </Button>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            {filteredModels.length === 0 ? (
              <p className="text-[10px] text-muted-foreground/60">
                {searchQuery ? 'No models match your search.' : 'No models registered. Load a model first.'}
              </p>
            ) : (
              <div className="space-y-1">
                {filteredModels.map(m => (
                  <div key={m.model_id} className="rounded-lg border border-border/40 px-2.5 py-2 text-[11px] hover:bg-muted/20 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="font-medium truncate">{m.model_id}</span>
                          <span className={cn('text-[9px] px-1.5 py-0.5 rounded-full', m.status === 'loaded' ? 'bg-success/10 text-success' : m.status === 'failed' ? 'bg-destructive/10 text-destructive' : 'bg-muted/40 text-muted-foreground/60')}>{m.status}</span>
                        </div>
                        {m.registered_at && (
                          <div className="text-[10px] text-muted-foreground/60 mt-0.5 font-mono tabular-nums">
                            Registered {new Date(m.registered_at).toLocaleDateString()}
                          </div>
                        )}
                      </div>
                      {m.status === 'failed' && m.metrics && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 text-[10px]"
                          onClick={() => setExpandedModel(expandedModel === m.model_id ? null : m.model_id)}
                        >
                          {expandedModel === m.model_id ? 'Hide' : 'Details'}
                        </Button>
                      )}
                    </div>
                    {expandedModel === m.model_id && m.metrics && (
                      <div className="mt-1.5 rounded-lg bg-muted/20 p-2">
                        <div className="text-[9px] font-medium text-muted-foreground/60 uppercase tracking-wider mb-0.5">Error Details</div>
                        <div className="grid grid-cols-2 gap-0.5">
                          {Object.entries(m.metrics).map(([k, v]) => (
                            <div key={k} className="text-[10px]">
                              <span className="text-muted-foreground/60">{k.replace(/_/g, ' ')}: </span>
                              <span className="font-mono">{String(v)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {bestModel && Object.keys(bestModel).filter(k => !k.startsWith('_')).length > 0 && (
          <Card>
            <CardHeader className="pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-[11px] font-medium">Best Model</CardTitle>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
                {Object.entries(bestModel).filter(([k]) => !k.startsWith('_')).map(([key, value]) => (
                  <div key={key} className="rounded-lg bg-muted/20 p-2 text-center">
                    <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider capitalize">{key.replace(/_/g, ' ')}</div>
                    <div className="text-[11px] font-mono font-medium tabular-nums mt-0.5">{String(value ?? '—')}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
    </PageContainer>
  )
}
