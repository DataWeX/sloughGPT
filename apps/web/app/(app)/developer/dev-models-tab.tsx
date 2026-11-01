'use client'

/** Models tab — catalog, quantization, downloads and engine status. */
import ModelCatalogCard from '@/components/models/ModelCatalogCard'
import QuantizationCard from '@/components/models/QuantizationCard'
import DownloadsCard from '@/components/models/DownloadsCard'
import EngineStatusCard from '@/components/models/EngineStatusCard'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useModels } from '@/lib/cache/api-hooks'
import { useCallback } from 'react'

export function ModelsTab() {
  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useModels()
  const { healthLegacy: health } = useLiveStatus()
  const models = modelsData ?? []
  const isOnline = health !== null && health !== 'offline'
  const activeRuntimeId =
    health !== null && health !== 'offline' && health.model_loaded ? health.model_type : null

  const handleModelLoaded = useCallback(async () => {
    await refetchModels()
  }, [refetchModels])

  return (
    <div className="space-y-3">
      <ModelCatalogCard
        models={models}
        modelsLoading={modelsLoading}
        activeRuntimeId={activeRuntimeId}
        onModelLoaded={handleModelLoaded}
      />
      <QuantizationCard isOnline={isOnline} />
      <DownloadsCard />
      <EngineStatusCard />
    </div>
  )
}
