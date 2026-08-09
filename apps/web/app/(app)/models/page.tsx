'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect, useMemo } from 'react'
import { extractErrorMessage } from '@/lib/error-utils'
import { useRouter } from 'next/navigation'
import type { ModelEntry } from '@/lib/types/models'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useToastStore } from '@/lib/toast-store'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString } from '@/lib/format-bytes'
import { modelDisplayName } from '@/lib/inference-display'
import { modelController } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import { benchmarkController, type BenchmarkResult } from '@/lib/benchmark-controller'
import ModelStatusCard from '@/components/models/ModelStatusCard'
import ComposableLayersCard from '@/components/models/ComposableLayersCard'
import PersonalitiesCard from '@/components/models/PersonalitiesCard'
import PersonalityProfileCard from '@/components/models/PersonalityProfileCard'
import ModelCatalogCard from '@/components/models/ModelCatalogCard'
import { FineTunedModelsCard } from '@/components/training/FineTunedModelsCard'
import ModelPlaygroundCard from '@/components/models/ModelPlaygroundCard'
import ModelCacheCard from '@/components/models/ModelCacheCard'
import ModelUsageCard from '@/components/models/ModelUsageCard'
import QuantizationCard from '@/components/models/QuantizationCard'
import ModelsCard from '@/components/compare/ModelsCard'
import ComparisonTableCard from '@/components/compare/ComparisonTableCard'
import SummaryCard from '@/components/compare/SummaryCard'
import dynamicNext from 'next/dynamic'
import {
  useModels,
  useSouls,
  useCheckpoints,
  useCurrentSoul,
  useSwitchSoul,
} from '@/lib/query/api-hooks'

const OutputComparisonCard = dynamicNext<{ models: ModelEntry[] }>(() => import('@/components/compare/OutputComparisonCard'), { ssr: false })
const VisualComparisonCard = dynamicNext(() => import('@/components/compare/VisualComparisonCard'), { ssr: false })

export default function ModelsPage() {
  const router = useRouter()
  const [switchingSoul, setSwitchingSoul] = useState<string | null>(null)
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const { healthLegacy: health, health: liveHealth } = useLiveStatus()
  const refreshHealth = useCallback(async () => {
    await modelController.getHealth()
  }, [])
  const addToast = useToastStore(s => s.addToast)

  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels, error: modelsError } = useModels()
  const { data: soulsData, isLoading: soulsLoading, refetch: refetchSouls, error: soulsError } = useSouls()
  const { data: currentSoulData, refetch: refetchCurrentSoul } = useCurrentSoul()
  const { data: checkpointsData, isLoading: checkpointsLoading, refetch: refetchCheckpoints } = useCheckpoints()
  const { mutateAsync: switchSoul } = useSwitchSoul()

  const models = modelsData ?? []
  const souls = soulsData?.souls ?? []
  const currentSoul = currentSoulData?.name ?? soulsData?.current_soul ?? null
  const checkpoints = checkpointsData?.checkpoints ?? []
  const activeCheckpoint = checkpointsData?.active_checkpoint ?? null
  const activeRuntimeId = health !== null && health !== 'offline' && health.model_loaded ? health.model_type : null

  const handleSwitchSoul = async (name: string, checkpointName?: string) => {
    setSwitchingSoul(name)
    try {
      await switchSoul({ name, checkpointName })
      addToast(checkpointName ? `${name} + ${checkpointName}` : name, 'success')
    } catch (err) {
      addToast(extractErrorMessage(err, 'Failed'), 'error')
    } finally {
      setSwitchingSoul(null)
    }
  }

  const handleSaveTraits = useCallback(async (weights: Record<string, Record<string, number>>) => {
    try {
      await soulsController.saveTraitWeights(weights)
      addToast('Personality updated', 'success')
      setTraitWeights(weights)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Failed to save traits'), 'error')
    }
  }, [addToast])

  const fetchTraitWeights = useCallback(async () => {
    try {
      const w = await soulsController.getTraitWeights()
      if (w && !('error' in w)) setTraitWeights(w)
    } catch { addToast('Could not load trait weights', 'info') }
  }, [addToast])

  const [refreshing, setRefreshing] = useState(false)
  const [cacheUsage, setCacheUsage] = useState<{ total_gb: number; model_count: number } | null>(null)
  const [compareResults, setCompareResults] = useState<Record<string, BenchmarkResult | null>>({})
  const [compareRunning, setCompareRunning] = useState<Set<string>>(new Set())
  const [compareLoading, setCompareLoading] = useState(true)

  const compareModels: ModelEntry[] = useMemo(() => {
    const healthObj = health && health !== 'offline' ? health : null
    return (models ?? []).map(m => ({
      id: m.id || m.name,
      name: (m.id || m.name).replace(/^hf\//, ''),
      loaded: m.loaded || (healthObj?.model_type?.includes(m.id || m.name) ?? false),
      sizeGb: m.size_gb,
    }))
  }, [models, health])

  useEffect(() => { setCompareLoading(modelsLoading) }, [modelsLoading])

  const runBenchmark = async (modelId: string) => {
    setCompareRunning(prev => new Set(prev).add(modelId))
    setCompareResults(prev => ({ ...prev, [modelId]: null }))
    try {
      const result = await benchmarkController.run({ model: modelId })
      setCompareResults(prev => ({ ...prev, [modelId]: result }))
    } catch {
      setCompareResults(prev => ({ ...prev, [modelId]: { error: 'Failed' } as BenchmarkResult }))
      addToast(`Benchmark failed for ${modelId}`, 'error')
    } finally { setCompareRunning(prev => { const n = new Set(prev); n.delete(modelId); return n }) }
  }

  const runAllBenchmarks = async () => { await Promise.allSettled(compareModels.map(m => runBenchmark(m.id))) }

  const clearCompareResult = (modelId: string) => setCompareResults(prev => { const n = { ...prev }; delete n[modelId]; return n })

  const completedCompareResults = useMemo(() => Object.entries(compareResults).filter(([, r]) => r !== null && !r.error) as [string, BenchmarkResult][], [compareResults])

  const bestMetrics: Record<string, number> = useMemo(() => {
    if (completedCompareResults.length === 0) return { throughput: 0, latency: Infinity, p95: Infinity, params: 0 }
    return {
      throughput: Math.max(...completedCompareResults.map(([, r]) => r.throughput_tokens_per_sec)),
      latency: Math.min(...completedCompareResults.map(([, r]) => r.inference_time_ms)),
      p95: Math.min(...completedCompareResults.map(([, r]) => r.latency_p95_ms ?? Infinity)),
      params: Math.max(...completedCompareResults.map(([, r]) => r.num_parameters)),
    }
  }, [completedCompareResults])

  const chartData = useMemo(() => completedCompareResults
    .map(([modelId, r]) => ({ name: compareModels.find(m => m.id === modelId)?.name || modelId, throughput: r.throughput_tokens_per_sec, latency: r.inference_time_ms, memory: r.memory_mb }))
    .sort((a, b) => b.throughput - a.throughput), [completedCompareResults, compareModels])

  const handleRefresh = async () => {
    setRefreshing(true)
    await Promise.allSettled([
      refetchModels(),
      refetchSouls(),
      refetchCurrentSoul(),
      refetchCheckpoints(),
      refreshHealth(),
      fetchTraitWeights(),
    ])
    setRefreshing(false)
    addToast('Refreshed', 'success')
  }

  useEffect(() => { fetchTraitWeights() }, [fetchTraitWeights])
  useEffect(() => { modelController.getCacheUsage().then(setCacheUsage).catch(() => /* cache info unavailable */ {}) }, [])

  const isOnline = health !== null && health !== 'offline'
  const subtitle = health === null ? 'Connecting...'
    : !isOnline ? 'API offline'
    : health.model_loaded ? `${modelDisplayName(health.model_type)} · running`
    : 'No model loaded'

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        className="items-start"
        left={<AppRouteHeaderLead title="Models & Personalities" subtitle={subtitle} />}
        right={
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => router.push('/compare')}>
              Compare
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={() => {
              const data = models.map(m => ({
                id: m.id,
                name: m.name,
                type: m.type,
                source: m.source,
                params: m.params,
                size_mb: m.size_mb,
                size_gb: m.size_gb,
                cached: m.cached,
                loaded: m.loaded,
              }))
              downloadJson(data, `models-export-${todayDateString()}.json`)
              addToast(`Exported ${models.length} models`, 'success')
            }}>
              Export
            </Button>
            <Button type="button" variant="secondary" size="sm" disabled={refreshing} onClick={handleRefresh}><IconRefresh className={`w-3.5 h-3.5 mr-1 ${refreshing ? 'animate-spin' : ''}`} /> {refreshing ? 'Refreshing...' : 'Refresh'}</Button>
          </div>
        }
      />

      <div className="space-y-4">
        {modelsError && models.length === 0 && (
          <Card>
            <CardContent className="py-4 flex items-center justify-between">
              <span className="text-sm text-destructive">Failed to load models</span>
              <Button size="sm" variant="outline" onClick={() => refetchModels()}>Retry</Button>
            </CardContent>
          </Card>
        )}
        {soulsError && souls.length === 0 && (
          <Card>
            <CardContent className="py-4 flex items-center justify-between">
              <span className="text-sm text-destructive">Failed to load personalities</span>
              <Button size="sm" variant="outline" onClick={() => refetchSouls()}>Retry</Button>
            </CardContent>
          </Card>
        )}
        <ModelStatusCard
          isOnline={isOnline}
          health={health}
          currentSoul={currentSoul}
          activeCheckpoint={activeCheckpoint}
          modelsCount={models.length}
          soulsCount={souls.length}
          checkpointsCount={checkpoints.length}
          modelsLoading={modelsLoading}
          soulsLoading={soulsLoading}
          checkpointsLoading={checkpointsLoading}
        />
        <ModelUsageCard
          inferenceCount={liveHealth?.inference_count ?? 0}
          requestCount={liveHealth?.request_count ?? 0}
          modelType={health && health !== 'offline' ? health.model_type : null}
          isOnline={isOnline}
        />
        <ComposableLayersCard
          modelsCount={models.length}
          soulsCount={souls.length}
          checkpoints={checkpoints}
        />
        <PersonalitiesCard
          souls={souls}
          soulsLoading={soulsLoading}
          checkpoints={checkpoints}
          checkpointsLoading={checkpointsLoading}
          currentSoul={currentSoul}
          activeCheckpoint={activeCheckpoint}
          switchingSoul={switchingSoul}
          onSwitch={handleSwitchSoul}
        />
        <PersonalityProfileCard
          traitWeights={traitWeights}
          currentSoulName={currentSoul}
          onTraitsSaved={handleSaveTraits}
          onTraitsChanged={fetchTraitWeights}
        />
        <ModelCatalogCard
          models={models}
          modelsLoading={modelsLoading}
          activeRuntimeId={activeRuntimeId}
          onModelLoaded={async () => { await refreshHealth(); await refetchModels() }}
        />
        <FineTunedModelsCard
          activeModelId={activeRuntimeId}
          onLoaded={async () => { await refreshHealth(); await refetchModels() }}
        />
        <ModelPlaygroundCard activeRuntimeId={activeRuntimeId} />
        <QuantizationCard isOnline={isOnline} />
        <ModelCacheCard
          cacheUsage={cacheUsage}
          health={health && health !== 'offline' ? health : null}
          onRefresh={() => modelController.getCacheUsage().then(setCacheUsage).catch(() => /* cache refresh failed */ {})}
        />

        {/* Comparison section */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Model Comparison</CardTitle>
                <p className="text-sm text-muted-foreground">Side-by-side benchmark results across models</p>
              </div>
              <Button variant="outline" size="sm" onClick={runAllBenchmarks} disabled={compareLoading || compareRunning.size > 0}>
                <IconRefresh className="h-3.5 w-3.5 mr-1" /> Benchmark all
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
            {completedCompareResults.length === 0 && compareRunning.size === 0 && !compareLoading ? (
              <div className="text-center py-8 text-sm text-muted-foreground">
                Run a benchmark on one or more models to see comparison results here.
              </div>
            ) : (
              <>
            <ModelsCard models={compareModels} loading={compareLoading} results={compareResults} running={compareRunning} onBenchmark={runBenchmark} onClear={clearCompareResult} />
            <ComparisonTableCard completedResults={completedCompareResults} models={compareModels} bestMetrics={bestMetrics} />
            <SummaryCard completedResults={completedCompareResults} models={compareModels} />
            <OutputComparisonCard models={compareModels} />
            <VisualComparisonCard chartData={chartData} />
              </>
            )}
          </div>
        </CardContent>
      </Card>
      </div>


    </div>
  )
}
