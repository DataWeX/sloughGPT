'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { cn, Card, CardContent, CardHeader, CardTitle, Progress } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { catalogIdMatchesRuntime } from '@/lib/inference-display'
import { generateController } from '@/lib/generate-controller'
import { useLoadModel } from '@/lib/query/api-hooks'
import { useToastStore } from '@/lib/toast-store'
import { useConversionStatus, formatStage } from '@/hooks/useConversionStatus'
import { logger } from '@/lib/dev-log'

interface Model {
  id: string; name?: string; source?: string; description?: string; tags?: string[]
  size_mb?: number; size_gb?: number; params?: string; cached?: boolean; thumbnail?: string
}

interface ModelCatalogCardProps {
  models: Model[]
  modelsLoading: boolean
  activeRuntimeId: string | null
  onModelLoaded: () => Promise<void>
}

export default function ModelCatalogCard({ models, modelsLoading, activeRuntimeId, onModelLoaded }: ModelCatalogCardProps) {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const { mutateAsync: loadModel } = useLoadModel()
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [warmingModel, setWarmingModel] = useState<string | null>(null)
  const [modelSearch, setModelSearch] = useState('')
  const [trackedModel, setTrackedModel] = useState<string | null>(null)
  const { status: conversionStatus } = useConversionStatus(trackedModel)

  const handleLoadModel = async (modelId: string) => {
    setLoadingModel(modelId)
    setTrackedModel(modelId)
    try {
      const data = await loadModel(modelId)
      if (data.error) { addToast(data.error, 'error'); return }
      addToast(`${data.model_id ?? modelId} ready`, 'success')
      setLoadingModel(null)
      setWarmingModel(modelId)
      try { await generateController.generate({ prompt: 'Hello', max_new_tokens: 2 }) } catch {
        logger.warning('ModelCatalogCard: warmup generation failed', { modelId })
      }
      setWarmingModel(null)
    } catch (err) {
      addToast(err instanceof Error ? err.message : String(err), 'error')
    } finally {
      setLoadingModel(null)
      setWarmingModel(null)
      setTrackedModel(null)
      await onModelLoaded()
    }
  }

  const filtered = modelSearch
    ? models.filter(m => (m.name || m.id).toLowerCase().includes(modelSearch.toLowerCase()) || m.id.toLowerCase().includes(modelSearch.toLowerCase()))
    : models

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Model Catalog</CardTitle></CardHeader>
      <CardContent>
        {modelsLoading ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {[1,2,3,4,5,6].map(i => (
              <div key={i} className="animate-pulse rounded-lg bg-muted/50 border border-border/50 p-4 h-20" />
            ))}
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-8 text-sm text-muted-foreground">No models available</div>
        ) : (
          <>
            {models.length > 3 && (
              <div className="mb-3">
                <input
                  type="text"
                  value={modelSearch}
                  onChange={e => setModelSearch(e.target.value)}
                  placeholder="Search models..."
                  className="h-8 w-full max-w-xs rounded-md border border-border/60 bg-background px-2 text-xs placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
                />
              </div>
            )}
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {filtered.map((model) => {
                const isLoading = loadingModel === model.id
                const isLoaded = activeRuntimeId ? catalogIdMatchesRuntime(model.id, activeRuntimeId) : false
                return (
                  <div key={model.id}>
                  <div
                    className={cn("flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:-translate-y-0.5 hover:shadow-sm",
                      isLoaded ? "border-primary/40 bg-primary/5" : "border-border/60",
                      isLoading && "animate-pulse border-primary/30 bg-primary/[0.08]")}
                    onClick={() => router.push(`/model/${encodeURIComponent(model.id)}`)}
                    onKeyDown={(e) => { if (e.key === 'Enter') router.push(`/model/${encodeURIComponent(model.id)}`) }}
                    tabIndex={0}
                    role="button"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium truncate">{model.name || model.id}</div>
                      {model.description && (
                        <div className="text-[10px] text-muted-foreground/70 truncate mt-0.5">{model.description}</div>
                      )}
                      <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                        {model.params && <span className="text-[10px] text-muted-foreground font-mono">{model.params}</span>}
                        {model.size_gb && <span className="text-[10px] text-muted-foreground">{(model.size_gb).toFixed(1)} GB</span>}
                        {model.source && model.source !== 'local' && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{model.source}</span>
                        )}
                        {isLoaded && <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Loaded</span>}
                        {!isLoaded && model.cached && <span className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success font-medium">Cached</span>}
                        {model.tags && model.tags.slice(0, 2).map(tag => (
                          <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground font-medium">{tag}</span>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-1 shrink-0 ml-2">
                      {model.source === 'local' ? (
                        <span className="text-[10px] text-muted-foreground">Local</span>
                      ) : (
                        <Button size="sm" variant={isLoaded ? 'outline' : 'default'} className="h-7 text-xs px-3"
                          disabled={isLoading || warmingModel === model.id}
                          onClick={(e) => { e.stopPropagation(); handleLoadModel(model.id) }}>
                          {isLoading
                            ? (conversionStatus && conversionStatus.stage !== 'idle'
                                ? `${Math.round(conversionStatus.progress * 100)}%`
                                : <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />)
                            : warmingModel === model.id ? 'Warming…' : isLoaded ? 'Loaded' : 'Load'}
                        </Button>
                      )}
                    </div>
                  </div>
                  {/* Conversion progress bar */}
                  {isLoading && conversionStatus && conversionStatus.stage !== 'idle' && conversionStatus.stage !== 'ready' && (
                    <div key={`${model.id}-progress`} className="col-span-full mt-1">
                      <Progress
                        value={conversionStatus.progress * 100}
                        size="xs"
                        variant={conversionStatus.stage === 'error' ? 'error' : 'default'}
                        label={formatStage(conversionStatus.stage)}
                        showValue
                      />
                      <span className="text-[9px] text-muted-foreground ml-auto">{conversionStatus.elapsed_s.toFixed(0)}s</span>
                    </div>
                  )}
                  </div>
                )
              })}
            </div>
            {filtered.length === 0 && modelSearch && (
              <div className="text-center py-4 text-xs text-muted-foreground">No models matching &quot;{modelSearch}&quot;</div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
