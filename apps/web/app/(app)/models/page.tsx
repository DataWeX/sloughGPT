'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useToastStore } from '@/lib/toast-store'
import { modelDisplayName } from '@/lib/inference-display'
import { modelController } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import ModelStatusCard from '@/components/models/ModelStatusCard'
import ComposableLayersCard from '@/components/models/ComposableLayersCard'
import PersonalitiesCard from '@/components/models/PersonalitiesCard'
import PersonalityProfileCard from '@/components/models/PersonalityProfileCard'
import ModelCatalogCard from '@/components/models/ModelCatalogCard'
import ModelPlaygroundCard from '@/components/models/ModelPlaygroundCard'
import ModelCacheCard from '@/components/models/ModelCacheCard'
import QuantizationCard from '@/components/models/QuantizationCard'
import {
  useModels,
  useSouls,
  useCheckpoints,
  useCurrentSoul,
  useSwitchSoul,
} from '@/lib/query/api-hooks'

export default function ModelsPage() {
  const [switchingSoul, setSwitchingSoul] = useState<string | null>(null)
  const [traitWeights, setTraitWeights] = useState<Record<string, any> | null>(null)
  const { healthLegacy: health } = useLiveStatus()
  const refreshHealth = useCallback(async () => {}, [])
  const addToast = useToastStore(s => s.addToast)

  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useModels()
  const { data: soulsData, isLoading: soulsLoading, refetch: refetchSouls } = useSouls()
  const { data: currentSoulData, isLoading: currentSoulLoading, refetch: refetchCurrentSoul } = useCurrentSoul()
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
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to save traits', 'error')
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

  useEffect(() => { fetchTraitWeights(); modelController.getCacheUsage().then(setCacheUsage).catch(() => {}) }, [fetchTraitWeights])

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
        right={<Button type="button" variant="secondary" size="sm" disabled={refreshing} onClick={handleRefresh}><IconRefresh className={`w-3.5 h-3.5 mr-1 ${refreshing ? 'animate-spin' : ''}`} /> {refreshing ? 'Refreshing...' : 'Refresh'}</Button>}
      />

      <div className="space-y-4">
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
        <ModelPlaygroundCard activeRuntimeId={activeRuntimeId} />
        <QuantizationCard isOnline={isOnline} />
        <ModelCacheCard
          cacheUsage={cacheUsage}
          health={health}
          onRefresh={() => modelController.getCacheUsage().then(setCacheUsage).catch(() => {})}
        />
      </div>


    </div>
  )
}
