'use client'

import { useState, useEffect, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { IconRefresh } from '@/components/ui'
import { cn } from '@/lib/cn'
import { useToastStore } from '@/lib/toast-store'
import { visualController } from '@/lib/controllers'
import StatusCard from '@/components/visual/StatusCard'
import ImageInferenceCard from '@/components/visual/ImageInferenceCard'
import CreateVisualDatasetCard from '@/components/visual/CreateVisualDatasetCard'
import VideoInferenceCard from '@/components/visual/VideoInferenceCard'
import TrainVisualAICard from '@/components/visual/TrainVisualAICard'
import VideoTrainingCard from '@/components/visual/VideoTrainingCard'
import LoadVisualModelCard from '@/components/visual/LoadVisualModelCard'
import SavedCheckpointsCard from '@/components/visual/SavedCheckpointsCard'
import PDFAnalysisCard from '@/components/visual/PDFAnalysisCard'

export default function VisualPage() {
  const addToast = useToastStore(s => s.addToast)

  // ── State ──────────────────────────────────────────────────────
  const [visualStatus, setVisualStatus] = useState<any>(null)
  const [dpoStatus, setDpoStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)



  // ── Load ───────────────────────────────────────────────────────
  const [loadModelDir, setLoadModelDir] = useState('models/visual-finetuned')

  // ── Checkpoints ────────────────────────────────────────────────
  const [checkpoints, setCheckpoints] = useState<Array<{
    name: string; path: string; size_mb: number; final_loss: number | null;
    total_steps: number; vision_encoder?: string; llm?: string
  }>>([])
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(false)

  // ── Fetch all status ───────────────────────────────────────────
  const fetchAll = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true)
    try {
      const [vs, ds] = await Promise.all([
        visualController.getVisualStatus().catch(() => null),
        visualController.getDPOStatus().catch(() => null),
      ])
      setVisualStatus(vs as any)
      setDpoStatus(ds)
    } catch {}
    setLoading(false)
    setRefreshing(false)
  }, [])

  const fetchCheckpoints = useCallback(async () => {
    setLoadingCheckpoints(true)
    try {
      const result = await visualController.listCheckpoints()
      setCheckpoints(result)
    } catch {} finally {
      setLoadingCheckpoints(false)
    }
  }, [])

  useEffect(() => { fetchAll(); fetchCheckpoints() }, [fetchAll, fetchCheckpoints])

  // ── Handlers ───────────────────────────────────────────────────

  const handleLoadCheckpoint = async (name: string) => {
    try {
      await visualController.loadCheckpoint(name)
      addToast(`Checkpoint '${name}' loaded`, 'success')
      fetchAll(); fetchCheckpoints()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    }
  }

  const handleDeleteCheckpoint = async (name: string) => {
    try {
      await visualController.deleteCheckpoint(name)
      addToast(`Checkpoint '${name}' deleted`, 'success')
      fetchCheckpoints()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    }
  }

  const handleImportFromCheckpoint = (path: string) => {
    setLoadModelDir(path)
    addToast('Model directory set from checkpoint', 'info')
  }

  // ── Render ─────────────────────────────────────────────────────

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <AppRouteHeaderLead
            title="Visual AI"
            subtitle="Load, train, and run visual inference on images"
          />
        }
        right={
          <Button
            variant="ghost"
            size="sm"
            onClick={() => fetchAll(true)}
            disabled={refreshing}
          >
            <IconRefresh className={cn("h-4 w-4", refreshing && "animate-spin")} />
          </Button>
        }
      />

      <div className="space-y-4">
        <StatusCard loading={loading} visualStatus={visualStatus} dpoStatus={dpoStatus} />
        <ImageInferenceCard />
        <CreateVisualDatasetCard onCreated={fetchAll} />

        <TrainVisualAICard />
        <VideoTrainingCard />

        <VideoInferenceCard />

        <LoadVisualModelCard onLoaded={fetchAll} />
        <SavedCheckpointsCard
          loading={loadingCheckpoints}
          checkpoints={checkpoints}
          onUsePath={handleImportFromCheckpoint}
          onLoad={handleLoadCheckpoint}
          onDelete={handleDeleteCheckpoint}
        />

        <PDFAnalysisCard />
      </div>
    </div>
  )
}
