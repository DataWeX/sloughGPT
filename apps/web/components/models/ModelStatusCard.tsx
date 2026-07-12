'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Chip } from '@sloughgpt/strui'
import { KpiGrid, StatCard } from '@sloughgpt/strui'
import { modelDisplayName } from '@/lib/inference-display'
import { cn } from '@/lib/cn'

interface ModelStatusCardProps {
  isOnline: boolean
  health: any
  currentSoul: string | null
  activeCheckpoint: string | null
  modelsCount: number
  soulsCount: number
  checkpointsCount: number
  modelsLoading: boolean
  soulsLoading: boolean
  checkpointsLoading: boolean
}

export default function ModelStatusCard({
  isOnline, health, currentSoul, activeCheckpoint,
  modelsCount, soulsCount, checkpointsCount,
  modelsLoading, soulsLoading, checkpointsLoading,
}: ModelStatusCardProps) {
  const quant = health?.quantization?.quantized ? health.quantization : null
  return (
    <>
      {isOnline && (
        <Card>
          <CardHeader><CardTitle className="text-base">Active Pipeline</CardTitle></CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-2">
                <span className={cn("w-2 h-2 rounded-full", health.model_loaded ? "bg-success animate-pulse" : "bg-warning")} />
                <span className="text-xs font-medium">{modelDisplayName(health.model_type) || 'No model'}</span>
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
              {health.inference_count != null && (
                <span className="ml-auto text-[10px] text-muted-foreground">{health.inference_count} inferences</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <KpiGrid columns={3}>
        <StatCard label="Models" value={modelsLoading ? '—' : modelsCount.toString()} icon={<span className="text-xs font-mono">M</span>} />
        <StatCard label="Personalities" value={soulsLoading ? '—' : soulsCount.toString()} icon={<span className="text-xs">🎭</span>} />
        <StatCard label="Checkpoints" value={checkpointsLoading ? '—' : checkpointsCount.toString()} icon={<span className="text-xs">📦</span>} />
      </KpiGrid>
    </>
  )
}
