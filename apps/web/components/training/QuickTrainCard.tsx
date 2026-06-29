'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { DatasetImportModal } from '@/components/DatasetImportModal'
import { trainingController } from '@/lib/controllers'
import { datasetController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { Dataset } from '@/lib/dataset-controller'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

function datasetLabel(ds: Dataset): string {
  const size = ds.size != null ? `${(ds.size / 1024).toFixed(1)} KB` : ''
  if (ds.type === 'vlm' && ds.vlm_metadata) {
    return `${ds.name} (VLM: ${ds.vlm_metadata.image_count} images, ${size})`
  }
  const suffix = ds.samples && ds.samples > 0 ? ` (${ds.samples.toLocaleString()} samples, ${size})` : ` (${size})`
  return `${ds.name}${suffix}`
}

export function QuickTrainCard({
  datasets,
  checkpoints,
}: {
  datasets: UseTrainingDatasetsReturn
  checkpoints: UseTrainingCheckpointsReturn
}) {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const qtTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [quickDataset, setQuickDataset] = useState('')
  const [quickStats, setQuickStats] = useState<import('@/lib/dataset-controller').DatasetStats | null>(null)
  const [quickExplanation, setQuickExplanation] = useState('')
  const [quickJobId, setQuickJobId] = useState('')
  const [quickStatusMessage, setQuickStatusMessage] = useState('')
  const [quickTraining, setQuickTraining] = useState(false)
  const [quickComplete, setQuickComplete] = useState(false)

  useEffect(() => {
    if (quickDataset) {
      datasetController.getStats(quickDataset).then(setQuickStats).catch(() => setQuickStats(null))
    } else {
      setQuickStats(null)
    }
  }, [quickDataset])

  useEffect(() => {
    return () => {
      if (qtTimeoutRef.current) { clearTimeout(qtTimeoutRef.current); qtTimeoutRef.current = null }
    }
  }, [])

  const reset = () => {
    setQuickComplete(false)
    setQuickExplanation('')
    setQuickDataset('')
    setQuickStats(null)
  }

  const startQuickTrain = async () => {
    if (!quickDataset) { addToast('Pick a dataset first', 'error'); return }
    setQuickTraining(true)
    setQuickComplete(false)
    setQuickStatusMessage('Setting up training...')
    try {
      const res = await trainingController.startQuick({ dataset: quickDataset })
      setQuickJobId(res.job_id)
      setQuickExplanation(res.explanation)
      setQuickStatusMessage('Training started — this may take a few minutes...')

      let retries = 0
      const poll = async () => {
        try {
          const jobs = await trainingController.list()
          const job = jobs.find((j: import('@/lib/training-controller').TrainingJob) => j.id === res.job_id)
          if (!job) return
          if (job.message) setQuickStatusMessage(job.message)
          if (job.status === 'completed') {
            setQuickTraining(false)
            setQuickComplete(true)
            setQuickExplanation(job.explanation || res.explanation)
            checkpoints.fetchCheckpoints()
            checkpoints.fetchJobs()
            addToast('Training complete!', 'success')
          } else if (job.status === 'failed') {
            setQuickTraining(false)
            addToast(job.message || 'Training failed', 'error')
          } else {
            retries = 0
            qtTimeoutRef.current = setTimeout(poll, 3000)
          }
        } catch {
          retries += 1
          if (retries >= 5) {
            setQuickTraining(false)
            addToast('Lost connection to training server', 'error')
            return
          }
          qtTimeoutRef.current = setTimeout(poll, 5000)
        }
      }
      qtTimeoutRef.current = setTimeout(poll, 2000)
    } catch (err: unknown) {
      setQuickTraining(false)
      addToast('Something went wrong starting training', 'error')
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Quick Train</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Pick a dataset, hit train — we figure out the rest.
        </p>
        <div className="flex items-center gap-2">
          {datasets.datasets.length === 0 ? (
            <>
              <span className="text-xs text-muted-foreground">No datasets yet — import one to get started.</span>
              <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>
                + Import
              </Button>
            </>
          ) : (
            <>
              <select
                value={quickDataset}
                onChange={e => setQuickDataset(e.target.value)}
                className="h-8 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground flex-1 max-w-sm"
                disabled={quickTraining}
                aria-label="Dataset for quick training"
              >
                <option value="">Select a dataset...</option>
                {datasets.datasets.map(ds => (
                  <option key={ds.id} value={ds.id}>{datasetLabel(ds)}</option>
                ))}
              </select>
              <Button size="sm" disabled={!quickDataset || quickTraining} onClick={startQuickTrain}>
                {quickTraining ? 'Training...' : 'Start training'}
              </Button>
            </>
          )}
        </div>

        <DatasetImportModal
          open={datasets.importModalOpen}
          onOpenChange={datasets.setImportModalOpen}
          onImportComplete={(datasetId: string) => {
            void datasets.fetchDatasets().then(() => datasets.setSelectedDataset(datasetId))
          }}
        />

        {quickExplanation && !quickTraining && !quickComplete && (
          <p className="text-xs text-muted-foreground/80 italic">{quickExplanation}</p>
        )}
        {quickStats && !quickTraining && !quickComplete && (
          <>
            <p className="text-xs text-muted-foreground/60">
              Detected: {quickStats.format} ({quickStats.samples} samples, {Math.round(quickStats.avg_length || 0)} avg chars)
            </p>
            {(quickStats.samples < 5 || (quickStats.avg_length || 0) < 50 || quickStats.format === 'text') && (
              <div className="rounded border border-amber-500/20 bg-amber-500/5 p-2 space-y-1">
                {quickStats.samples < 5 && (
                  <p className="text-[11px] text-amber-600 dark:text-amber-400">⚠ Very few samples — training may overfit. Add more data for better results.</p>
                )}
                {(quickStats.avg_length || 0) < 50 && quickStats.samples >= 5 && (
                  <p className="text-[11px] text-amber-600 dark:text-amber-400">⚠ Very short text — longer samples help the model learn patterns.</p>
                )}
                {quickStats.format === 'text' && (
                  <p className="text-[11px] text-muted-foreground">Tip: Conversation format (JSONL with messages) trains better than plain text.</p>
                )}
              </div>
            )}
          </>
        )}

        {quickTraining && (
          <div className="space-y-2" role="status" aria-live="polite" aria-label="Training progress">
            <p className="text-sm text-muted-foreground">{quickStatusMessage}</p>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-primary animate-pulse rounded-full" style={{ width: '40%' }} />
            </div>
          </div>
        )}

        {quickComplete && (
          <div className="rounded-lg border border-success/20 bg-success/5 p-3 space-y-2">
            <p className="text-sm font-medium text-success">Training complete!</p>
            {quickExplanation && <p className="text-xs text-muted-foreground">{quickExplanation}</p>}
            <div className="flex flex-wrap gap-2">
              <Button size="sm" onClick={() => router.push('/chat')}>Try in chat</Button>
              <Button size="sm" variant="ghost" onClick={reset}>
                Train another
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
