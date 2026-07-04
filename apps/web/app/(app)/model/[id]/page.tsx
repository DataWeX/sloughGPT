'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/tags'
import { StatCard, KpiGrid, Skeleton } from '@/components/ui/display'
import { Slider } from '@/components/ui/slider'
import { IconRefresh, IconTrash, IconCheck, IconCopy } from '@/components/ui'
import { cn } from '@/lib/cn'
import { modelController, type ModelInfo, type HealthStatus } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult } from '@/lib/benchmark-controller'
import { generationConfigController, type GenerationConfig } from '@/lib/generation-config-controller'
import { useToastStore } from '@/lib/toast-store'
import { apiGet } from '@/lib/http-client'

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
  const [modelLogs, setModelLogs] = useState<any[]>([])
  const [loadState, setLoadState] = useState<'idle' | 'loading' | 'loaded' | 'error'>('idle')
  const [uptime, setUptime] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval>>()
  const [genConfig, setGenConfig] = useState<GenerationConfig>({
    temperature: 0.7,
    max_new_tokens: 256,
    top_p: 1.0,
    top_k: 50,
  })
  const [configLoading, setConfigLoading] = useState(true)
  const [configSaving, setConfigSaving] = useState(false)

  const fetchData = useCallback(async () => {
    try {
      const [models, h, logsRes] = await Promise.all([
        modelController.list(),
        modelController.getHealth(),
        apiGet<{ logs: string[] }>('/models/logs?limit=10').catch(() => ({ logs: [] })),
      ])
      setHealth(h)
      setModelLogs(logsRes.logs)
      const m = models.find(m => m.id === modelId || m.name === modelId) || null
      setModel(m)
      setLoadState(h?.model_loaded && (h.model_type?.includes(modelId) ?? false) ? 'loaded' : 'idle')
    } catch (e) {
      addToast('Something went wrong loading the model', 'error')
    } finally {
      setLoading(false)
    }
  }, [modelId, addToast])

  useEffect(() => {
    if (!modelId) { router.push('/models'); return }
    fetchData()
    generationConfigController.get().then(cfg => {
      if (cfg && typeof cfg.temperature === 'number') setGenConfig(cfg)
    }).catch(() => {}).finally(() => setConfigLoading(false))
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
      addToast(`Something went wrong loading ${modelId}`, 'error')
    }
  }

  const handleUnload = async () => {
    try {
      await modelController.unloadModel(modelId)
      setLoadState('idle')
      setHealth(prev => prev ? { ...prev, model_loaded: false, model_type: '' } : null)
      addToast('Model stopped', 'info')
    } catch {
      addToast('Something went wrong', 'error')
    }
  }

  const handleSaveConfig = async () => {
    setConfigSaving(true)
    try {
      await generationConfigController.update(genConfig)
      addToast('Generation config updated', 'success')
    } catch {
      addToast('Failed to save config', 'error')
    } finally {
      setConfigSaving(false)
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
      addToast('Performance test failed', 'error')
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
                           <IconTrash className="h-3 w-3 mr-1" /> Remove
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
                  {model?.cached !== undefined && <DetailItem label="Cached" value={model.cached ? 'Yes' : 'No'} />}
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

            {/* Quick test card */}
            {isLoaded && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Quick test</CardTitle>
                </CardHeader>
                <CardContent>
                  <ModelTestPrompt modelId={modelId} />
                </CardContent>
              </Card>
            )}

            {/* Generation Config card */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Generation Config</CardTitle>
                  <Button size="sm" className="h-7 text-xs" onClick={handleSaveConfig} disabled={configLoading || configSaving}>
                    {configSaving ? 'Saving…' : 'Save'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {configLoading ? (
                  <div className="space-y-3">
                    <Skeleton className="h-6" />
                    <Skeleton className="h-6" />
                    <Skeleton className="h-6" />
                    <Skeleton className="h-6" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <ConfigSlider label="Temperature" value={genConfig.temperature} min={0} max={2} step={0.1} onChange={v => setGenConfig(p => ({ ...p, temperature: v }))} />
                    <ConfigSlider label="Max tokens" value={genConfig.max_new_tokens} min={1} max={4096} step={1} onChange={v => setGenConfig(p => ({ ...p, max_new_tokens: v }))} />
                    <ConfigSlider label="Top-p" value={genConfig.top_p ?? 1} min={0} max={1} step={0.05} onChange={v => setGenConfig(p => ({ ...p, top_p: v }))} />
                    <ConfigSlider label="Top-k" value={genConfig.top_k ?? 50} min={0} max={200} step={1} onChange={v => setGenConfig(p => ({ ...p, top_k: v }))} />
                  </div>
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

            {/* Model activity logs */}
            {modelLogs.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-base">Recent Activity</CardTitle>
                <Button size="sm" variant="ghost" onClick={() => apiGet<{ logs: string[] }>('/models/logs?limit=10').then(r => setModelLogs(r.logs)).catch(() => {})}>
                  <IconRefresh className="h-3 w-3" />
                </Button>
              </CardHeader>
              <CardContent>
                <div className="space-y-0.5 max-h-32 overflow-y-auto">
                  {modelLogs.map((log: any, i: number) => (
                    <p key={i} className="text-[11px] font-mono text-muted-foreground/70 truncate">{typeof log === 'string' ? log : log.message || JSON.stringify(log)}</p>
                  ))}
                </div>
              </CardContent>
            </Card>
            )}
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

function ConfigSlider({ label, value, min, max, step, onChange }: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs text-muted-foreground">{label}</label>
        <span className="text-xs font-mono tabular-nums">{value}</span>
      </div>
      <Slider value={[value]} onValueChange={([v]) => onChange(v)} min={min} max={max} step={step} />
    </div>
  )
}

function ModelTestPrompt({ modelId }: { modelId: string }) {
  const [prompt, setPrompt] = useState('Hello, who are you?')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const handleTest = useCallback(async () => {
    if (!prompt.trim() || loading) return
    setLoading(true)
    setOutput('')
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/inference/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), max_new_tokens: 200 }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setOutput(data.text || data.response || JSON.stringify(data))
    } catch (err) {
      addToast('Test failed — is the model loaded?', 'error')
    } finally {
      setLoading(false)
    }
  }, [prompt, loading, addToast])

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleTest() }}
          placeholder="Type a prompt to test..."
          className="flex-1 h-8 rounded-md border border-border/60 bg-background px-3 text-sm"
          disabled={loading}
        />
        <Button size="sm" onClick={handleTest} disabled={loading || !prompt.trim()}>
          {loading ? 'Testing…' : 'Test'}
        </Button>
      </div>
      {output && (
        <div className="rounded-md border border-border/40 bg-muted/20 p-3 text-sm text-muted-foreground whitespace-pre-wrap max-h-48 overflow-y-auto">
          {output}
        </div>
      )}
    </div>
  )
}
