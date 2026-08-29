'use client'

import { memo } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Chip, Skeleton } from '@sloughgpt/strui'
import { KpiGrid, StatCard } from '@sloughgpt/strui'
import { IconModels, IconBrain, IconTraining } from '@sloughgpt/strui'
import { modelDisplayName } from '@/lib/inference-display'
import type { HealthStatus } from '@/lib/model-controller'

interface ModelStatusCardProps {
  isOnline: boolean
  health: HealthStatus | 'offline' | null
  currentSoul: string | null
  activeCheckpoint: string | null
  modelsCount: number
  soulsCount: number
  checkpointsCount: number
  modelsLoading: boolean
  soulsLoading: boolean
  checkpointsLoading: boolean
}

export default memo(function ModelStatusCard({
  isOnline, health, currentSoul, activeCheckpoint,
  modelsCount, soulsCount, checkpointsCount,
  modelsLoading, soulsLoading, checkpointsLoading,
}: ModelStatusCardProps) {
  const h = health && health !== 'offline' ? health : null
  const quant = h?.quantization?.quantized ? h.quantization : null
  return (
    <>
      {isOnline && h && (
        <Card>
          <CardHeader><CardTitle className="text-base">Active Pipeline</CardTitle></CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-2">
                <span className={cn("w-2 h-2 rounded-full", h.model_loaded ? "bg-success animate-pulse" : "bg-warning")} />
                <span className="text-xs font-medium">{modelDisplayName(h.model_type) || 'No model'}</span>
              </div>
              {quant && (
                <Chip label={`int${quant.bits ?? quant.summary?.bits ?? '?'} ${quant.mode === 'asymmetric' ? 'asym' : ''}`} />
              )}
              {currentSoul && (
                <>
                  <span className="text-muted-foreground/40 text-xs">→</span>
                  <Chip label={currentSoul} />
                </>
              )}
              {activeCheckpoint && (
                <>
                  <span className="text-muted-foreground/40 text-xs">→</span>
                  <Chip label={activeCheckpoint} />
                </>
              )}
              {h.inference_count != null && (
                <span className="ml-auto text-[10px] text-muted-foreground">{h.inference_count} inferences</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <KpiGrid columns={3}>
        <StatCard label="Models" value={modelsLoading ? <Skeleton className="h-5 w-8 inline-block" /> : modelsCount.toString()} icon={<IconModels className="h-3.5 w-3.5 text-primary" />} />
        <StatCard label="Personalities" value={soulsLoading ? <Skeleton className="h-5 w-8 inline-block" /> : soulsCount.toString()} icon={<IconBrain className="h-3.5 w-3.5 text-accent" />} />
        <StatCard label="Checkpoints" value={checkpointsLoading ? <Skeleton className="h-5 w-8 inline-block" /> : checkpointsCount.toString()} icon={<IconTraining className="h-3.5 w-3.5 text-success" />} />
      </KpiGrid>
    </>
  )
})
