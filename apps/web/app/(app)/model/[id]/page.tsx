'use client'
import { logger } from '@/lib/dev-log'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton, KeyValueList, StatusDot, SettingsRow } from '@sloughgpt/strui'
import { Slider } from '@sloughgpt/strui'
import { IconRefresh, IconTrash, IconCheck, IconCopy } from '@sloughgpt/strui'
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
    }).catch((e) => logger.debug('Config load failed', e)).finally(() => setConfigLoading(false))
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
                      variant={isLoaded ? 'success' as const : loadState === 'error' ? 'error' as const : 'warning' as const}
                      size="sm"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    {isLoaded ? (
                      <>
                        <Button size="sm" className="h-7 text-xs" onClick={() => router.push('/chat')}>
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
                <KeyValueList
                  dense
                  items={[
                    { label: 'Model ID', value: modelId, mono: true },
                    { label: 'Source', value: model?.source || 'huggingface' },
                    { label: 'Device', value: health?.device || '—' },
                    ...(model?.size_gb ? [{ label: 'Size', value: `${model.size_gb.toFixed(2)} GB` }] : []),
                    ...(model?.cached !== undefined ? [{ label: 'Cached', value: model.cached ? 'Yes' : 'No' }] : []),
                    ...(uptime && isLoaded ? [{ label: 'Uptime', value: uptime }] : []),
                    ...(health?.inference_count !== undefined ? [{ label: 'Inferences', value: health.inference_count.toString() }] : []),
                  ]}
                />
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
                  <div className="space-y-1 divide-y divide-border">
                    <SettingsRow title="Temperature" control={<span className="text-xs font-mono tabular-nums w-12 text-right">{genConfig.temperature}</span>}>
                    </SettingsRow>
                    <div className="px-1 py-2">
                      <Slider value={[genConfig.temperature]} onValueChange={([v]: number[]) => setGenConfig(p => ({ ...p, temperature: v }))} min={0} max={2} step={0.1} />
                    </div>
                    <SettingsRow title="Max tokens" control={<span className="text-xs font-mono tabular-nums w-12 text-right">{genConfig.max_new_tokens}</span>}>
                    </SettingsRow>
                    <div className="px-1 py-2">
                      <Slider value={[genConfig.max_new_tokens]} onValueChange={([v]: number[]) => setGenConfig(p => ({ ...p, max_new_tokens: v }))} min={1} max={4096} step={1} />
                    </div>
                    <SettingsRow title="Top-p" control={<span className="text-xs font-mono tabular-nums w-12 text-right">{genConfig.top_p ?? 1}</span>}>
                    </SettingsRow>
                    <div className="px-1 py-2">
                      <Slider value={[genConfig.top_p ?? 1]} onValueChange={([v]: number[]) => setGenConfig(p => ({ ...p, top_p: v }))} min={0} max={1} step={0.05} />
                    </div>
                    <SettingsRow title="Top-k" control={<span className="text-xs font-mono tabular-nums w-12 text-right">{genConfig.top_k ?? 50}</span>}>
                    </SettingsRow>
                    <div className="px-1 py-2">
                      <Slider value={[genConfig.top_k ?? 50]} onValueChange={([v]: number[]) => setGenConfig(p => ({ ...p, top_k: v }))} min={0} max={200} step={1} />
                    </div>
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
                <KeyValueList
                  dense
                  items={[
                    { label: 'Type', value: model?.type || health?.model_type || modelId },
                    { label: 'Source', value: model?.source || 'HuggingFace' },
                    ...(model?.params ? [{ label: 'Parameters (raw)', value: model.params }] : []),
                    ...(health?.vocab_size ? [{ label: 'Vocabulary size', value: health.vocab_size.toLocaleString() }] : []),
                    ...(health?.block_size ? [{ label: 'Block size (context)', value: health.block_size.toLocaleString() }] : []),
                    ...(health?.soul_engine_active ? [{ label: 'Soul engine', value: health.soul_name || 'active' }] : []),
                  ]}
                />
                {model?.tags && model.tags.length > 0 && (
                  <div className="mt-3">
                    <span className="text-xs text-muted-foreground block mb-1">Tags</span>
                    <div className="flex flex-wrap gap-1">
                      {model.tags.map(t => (
                        <Badge key={t} label={t} variant={"default" as const} size="sm" />
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Model activity logs */}
            {modelLogs.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-base">Recent Activity</CardTitle>
                <Button size="sm" variant="ghost" aria-label="Refresh activity logs" onClick={() => apiGet<{ logs: string[] }>('/models/logs?limit=10').then(r => setModelLogs(r.logs)).catch(() => /* activity log refresh failed */ {})}>
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
      const { generateController } = await import('@/lib/generate-controller')
      const data = await generateController.generate({ prompt: prompt.trim(), max_new_tokens: 200 })
      setOutput(data.text || '')
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
