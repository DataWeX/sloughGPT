'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { registryController, type RegisteredModel, type RegistryStats } from '@/lib/registry-controller'
import { useToastStore } from '@/lib/toast-store'

export default function RegistryPage() {
  const [models, setModels] = useState<RegisteredModel[]>([])
  const [stats, setStats] = useState<RegistryStats | null>(null)
  const [bestModel, setBestModel] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const addToast = useToastStore(s => s.addToast)

  const fetchData = async () => {
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
      addToast('Failed to load registry data', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Registry" subtitle="Model registry" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Registry" subtitle={`${models.length} models registered`} />} />
      <div className="space-y-4">
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: 'Total', value: String(stats.total_models ?? 0) },
              { label: 'Loaded', value: String(stats.loaded_models ?? 0), color: 'text-success' },
              { label: 'Failed', value: String(stats.failed_models ?? 0), color: stats.failed_models ? 'text-destructive' : '' },
              { label: 'Circuit Breaker', value: stats.circuit_breaker_open ? 'Open' : 'Closed', color: stats.circuit_breaker_open ? 'text-destructive' : 'text-success' },
            ].map(s => (
              <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                <div className="text-xs text-muted-foreground">{s.label}</div>
                <div className={`text-lg font-mono font-medium ${s.color ?? ''}`}>{s.value}</div>
              </div>
            ))}
          </div>
        )}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Registered Models</CardTitle>
            <Button size="sm" variant="ghost" onClick={fetchData}>
              <IconRefresh className="h-3.5 w-3.5" />
            </Button>
          </CardHeader>
          <CardContent>
            {models.length === 0 ? (
              <p className="text-sm text-muted-foreground">No models registered. Load a model first.</p>
            ) : (
              <div className="space-y-2">
                {models.map(m => (
                  <div key={m.model_id} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm hover:bg-muted/50 transition-colors">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">{m.model_id}</span>
                        <span className={`text-[10px] px-1 rounded ${
                          m.status === 'loaded' ? 'bg-success/10 text-success' :
                          m.status === 'failed' ? 'bg-destructive/10 text-destructive' :
                          'bg-muted text-muted-foreground'
                        }`}>{m.status}</span>
                      </div>
                      {m.registered_at && (
                        <div className="text-xs text-muted-foreground mt-0.5">
                          Registered {new Date(m.registered_at).toLocaleDateString()}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {bestModel && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Best Model</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {Object.entries(bestModel).filter(([k]) => !k.startsWith('_')).map(([key, value]) => (
                  <div key={key} className="rounded-md bg-muted/30 p-3 text-center">
                    <div className="text-xs text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</div>
                    <div className="text-sm font-mono font-medium">{String(value ?? '—')}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
