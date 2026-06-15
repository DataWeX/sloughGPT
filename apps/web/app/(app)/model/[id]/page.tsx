'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/tags'
import { StatCard, KpiGrid, Skeleton } from '@/components/ui/display'
import { IconRefresh, IconTrash, IconCheck, IconCopy } from '@/components/ui'
import { cn } from '@/lib/cn'
import { modelController, type ModelInfo, type HealthStatus } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult } from '@/lib/benchmark-controller'
import { useToastStore } from '@/lib/toast-store'

export default function ModelDetailPage() {
  const params = useParams()
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const modelId = decodeURIComponent((params.id as string) || '')

  const [model, setModel] = useState<ModelInfo | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null)
  const [benchmarking, setBenchmarking] = useState(false)
  const [loadState, setLoadState] = useState<'idle' | 'loading' | 'loaded' | 'error'>('idle')
  const [uptime, setUptime] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval>>()

  const fetchData = useCallback(async () => {
    try {
      const [models, h] = await Promise.all([
        modelController.list(),
        modelController.getHealth(),
      ])
      setHealth(h)
      const m = models.find(m => m.id === modelId || m.name === modelId) || null
      setModel(m)
      setLoadState(h?.model_loaded && (h.model_type?.includes(modelId) ?? false) ? 'loaded' : 'idle')
    } catch (e) {
      addToast('Failed to load model data', 'error')
    } finally {
      setLoading(false)
    }
  }, [modelId, addToast])

  useEffect(() => {
    if (!modelId) { router.push('/models'); return }
    fetchData()
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [modelId, fetchData, router])

  useEffect(() => {
    if (loadState === 'loaded') {
      const start = Date.now()
      intervalRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - start) / 1000)
        const h = Math.floor(elapsed / 3600)
        const m = Math.floor((elapsed % 3600) / 60)
        const s = elapsed % 60
        setUptime(h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`)
      }, 1000)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [loadState])

  const handleLoad = async () => {
    setLoadState('loading')
    try {
      const result = await modelController.load(modelId)
      setLoadState('loaded')
      setHealth(prev => prev ? { ...prev, model_loaded: true, model_type: modelId, device: result.device || prev.device } : null)
      addToast(`Model ready: ${modelId} (${result.device || 'cpu'})`, 'success')
    } catch {
      setLoadState('error')
      addToast(`Failed to load ${modelId}`, 'error')
    }
  }

  const handleUnload = async () => {
    try {
      await modelController.unloadModel(modelId)
      setLoadState('idle')
      setHealth(prev => prev ? { ...prev, model_loaded: false, model_type: '' } : null)
      addToast('Model unloaded', 'info')
    } catch {
      addToast('Failed to unload', 'error')
    }
  }

  const runBenchmark = async () => {
    setBenchmarking(true)
    setBenchmark(null)
    try {
      const result = await benchmarkController.run({ model: modelId })
      setBenchmark(result)
    } catch {
      setBenchmark({ error: 'Benchmark failed' } as BenchmarkResult)
      addToast('Benchmark failed', 'error')
    } finally {
      setBenchmarking(false)
    }
  }

  const formatParamCount = (params?: string | number) => {
    if (!params) return null
    const n = typeof params === 'string' ? parseFloat(params) : params
    if (isNaN(n)) return null
    if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
    if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`
    return n.toString()
  }

  if (!modelId) return null

  const isLoaded = loadState === 'loaded'
  const isThisModelLoaded = isLoaded

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={() => router.push('/models')} className="h-7 px-1.5 text-xs text-muted-foreground hover:text-foreground">
              ← Models
            </Button>
            <AppRouteHeaderLead title={model?.name || modelId} />
          </div>
        }
      />

      <div className="space-y-4">
        {loading ? (
          <div className="space-y-3">
            <Skeleton className="h-32 rounded-lg" />
            <Skeleton className="h-48 rounded-lg" />
          </div>
        ) : !model ? (
          <Card>
            <CardContent className="py-8 text-center">
              <p className="text-sm text-muted-foreground mb-3">Model &ldquo;{modelId}&rdquo; not found.</p>
              <Button size="sm" onClick={() => router.push('/models')}>Browse models</Button>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Status card */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CardTitle className="text-base">Status</CardTitle>
                    <Badge
                      label={isLoaded ? 'Loaded' : loadState === 'loading' ? 'Loading…' : loadState === 'error' ? 'Error' : 'Inactive'}
                      variant={isLoaded ? 'success' : loadState === 'error' ? 'error' : 'warning'}
                      size="sm"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    {isLoaded ? (
                      <>
                        <Button size="sm" className="h-7 text-xs" onClick={() => window.location.href = '/chat'}>
                          Chat with this model
                        </Button>
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleUnload}>
                          <IconTrash className="h-3 w-3 mr-1" /> Unload
                        </Button>
                      </>
                    ) : (
                      <Button size="sm" className="h-7 text-xs" onClick={handleLoad} disabled={loadState === 'loading'}>
                        {loadState === 'loading' ? 'Loading…' : 'Load model'}
                      </Button>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                  <DetailItem label="Model ID" value={modelId} />
                  <DetailItem label="Source" value={model?.source || 'huggingface'} />
                  <DetailItem label="Device" value={health?.device || '—'} />
                  {model?.size_gb && <DetailItem label="Size" value={`${model.size_gb.toFixed(2)} GB`} />}
                  {uptime && isLoaded && <DetailItem label="Uptime" value={uptime} />}
                  {health?.inference_count !== undefined && (
                    <DetailItem label="Inferences" value={health.inference_count.toString()} />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Metrics card */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Metrics</CardTitle>
                  <Button size="sm" variant="outline" className="h-7 text-xs" onClick={runBenchmark} disabled={benchmarking || !isLoaded}>
                    <IconRefresh className={cn("h-3 w-3 mr-1", benchmarking && "animate-spin")} />
                    {benchmarking ? 'Benchmarking…' : benchmark ? 'Rerun' : 'Run benchmark'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {!isLoaded ? (
                  <p className="text-sm text-muted-foreground text-center py-4">Load this model to see live metrics.</p>
                ) : benchmark?.error ? (
                  <p className="text-sm text-destructive text-center py-4">Benchmark failed: {benchmark.error}</p>
                ) : benchmark ? (
                  <div className="space-y-4">
                    <KpiGrid columns={4}>
                      <StatCard label="Parameters" value={formatParamCount(benchmark.num_parameters) || benchmark.num_parameters.toLocaleString()} />
                      <StatCard label="Memory" value={`${benchmark.memory_mb.toFixed(0)} MB`} />
                      <StatCard label="Throughput" value={`${benchmark.throughput_tokens_per_sec.toFixed(1)} tok/s`} />
                      <StatCard label="Avg latency" value={`${benchmark.inference_time_ms.toFixed(0)} ms`} />
                    </KpiGrid>
                    <KpiGrid columns={3}>
                      <StatCard label="P50 latency" value={`${(benchmark.latency_p50_ms ?? 0).toFixed(0)} ms`} />
                      <StatCard label="P95 latency" value={`${(benchmark.latency_p95_ms ?? 0).toFixed(0)} ms`} />
                      <StatCard label="P99 latency" value={`${(benchmark.latency_p99_ms ?? 0).toFixed(0)} ms`} />
                    </KpiGrid>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">Run a benchmark to see performance metrics.</p>
                )}
              </CardContent>
            </Card>

            {/* Details card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3">
                  <DetailItem label="Type" value={model?.type || health?.model_type || modelId} />
                  <DetailItem label="Source" value={model?.source || 'HuggingFace'} />
                  {model?.params && <DetailItem label="Parameters (raw)" value={model.params} />}
                  {health?.vocab_size && <DetailItem label="Vocabulary size" value={health.vocab_size.toLocaleString()} />}
                  {health?.block_size && <DetailItem label="Block size (context)" value={health.block_size.toLocaleString()} />}
                  {health?.soul_engine_active && (
                    <DetailItem label="Soul engine" value={health.soul_name || 'active'} />
                  )}
                  {model?.tags && model.tags.length > 0 && (
                    <div className="col-span-2">
                      <span className="text-xs text-muted-foreground block mb-1">Tags</span>
                      <div className="flex flex-wrap gap-1">
                        {model.tags.map(t => (
                          <Badge key={t} label={t} variant="default" size="sm" />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">{label}</p>
      <p className="text-sm font-medium truncate">{value}</p>
    </div>
  )
}
