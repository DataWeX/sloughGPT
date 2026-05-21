'use client'

import { useApiHealth } from '@/hooks/useApiHealth'
import { ModelStatusPill, type ModelStatus } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { cn } from '@/lib/cn'

export function LiveInferenceStatus() {
  const { state: health, refresh: refreshHealth } = useApiHealth()

  const h = health as any
  const isInferencing = h?.is_inferencing
  const inferenceCount = h?.inference_count || 0
  const modelLoaded = h?.model_loaded
  const modelType = h?.model_type
  const vocabSize = h?.vocab_size
  const blockSize = h?.block_size
  const numParameters = h?.num_parameters

  const handleReload = async () => {
    try {
      await modelController.getHealth()
      refreshHealth()
    } catch (e) {
      console.error('Failed to reload health:', e)
    }
  }

  let status: ModelStatus = 'no-model'
  if (!health || health === 'offline') {
    status = 'offline'
  } else if (modelLoaded) {
    status = 'loaded'
  }

  return (
    <div className="flex items-center gap-2">
      <ModelStatusPill
        status={status}
        modelName={modelType || undefined}
        vocabSize={vocabSize}
        blockSize={blockSize}
        numParameters={numParameters}
        size="sm"
        onClick={handleReload}
      />

      {/* Inference status */}
      {isInferencing && (
        <div className="flex items-center gap-1 text-xs text-primary">
          <div className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-60" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
          </div>
          <span>generating...</span>
        </div>
      )}

      {/* Inference count */}
      {inferenceCount > 0 && !isInferencing && (
        <div className="text-xs text-muted-foreground">
          {inferenceCount} inference{inferenceCount !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  )
}