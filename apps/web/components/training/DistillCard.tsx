'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input, Slider, Label } from '@/components/ui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select'
import { useToastStore } from '@/lib/toast-store'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'

import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'

interface DistillCardProps {
  datasets: UseTrainingDatasetsReturn
  onComplete?: () => void
}

interface JobStatus {
  status: string
  progress: number
  current_epoch?: number
  train_loss?: number
  loss_history?: { step: number; value: number; type: string }[]
  checkpoint?: string
  error?: string
}

export function DistillCard({ datasets: ds, onComplete }: DistillCardProps) {
  const addToast = useToastStore(s => s.addToast)
  const [teacherModel, setTeacherModel] = useState('gpt2')
  const selectedDataset = ds.selectedDataset
  const setSelectedDataset = ds.setSelectedDataset
  const [temperature, setTemperature] = useState(4.0)
  const [epochs, setEpochs] = useState(10)
  const [embedDim, setEmbedDim] = useState(64)
  const [nLayers, setNLayers] = useState(2)
  const [running, setRunning] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<JobStatus | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => () => {
    if (pollRef.current) { clearInterval(pollRef.current) }
  }, [])

  const startDistill = useCallback(async () => {
    if (!selectedDataset) { addToast('Select a dataset first', 'error'); return }
    setRunning(true)
    setStatus(null)
    try {
      const res = await trainingJobsController.startDistill({
        teacher_model: teacherModel,
        dataset: selectedDataset,
        temperature,
        epochs,
        embed_dim: embedDim,
        n_layers: nLayers,
      })
      setJobId(res.job_id)
      const poll = async () => {
        try {
          const jobs = await trainingJobsController.list()
          const job = jobs.find((j: TrainingJob) => j.id === res.job_id)
          if (job) {
            setStatus({
              status: job.status,
              progress: job.progress || 0,
              current_epoch: job.current_epoch,
              train_loss: job.train_loss,
              loss_history: job.loss_history,
              checkpoint: job.checkpoint,
              error: job.explanation || job.status_message,
            } as JobStatus)
            if (job.status === 'completed' || job.status === 'failed') {
              setRunning(false)
              if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
              if (job.status === 'completed') {
                addToast('Distillation complete!', 'success')
                onComplete?.()
              } else {
                addToast(`Distillation failed: ${job.explanation || job.status_message}`, 'error')
              }
            }
          }
        } catch { /* poll */ }
      }
      pollRef.current = setInterval(poll, 3000)
    } catch {
      setRunning(false)
      addToast('Failed to start distillation', 'error')
    }
  }, [teacherModel, selectedDataset, temperature, epochs, embedDim, nLayers, addToast, onComplete])

  const handleLoadCheckpoint = useCallback(async () => {
    if (!status?.checkpoint) return
    const name = status.checkpoint.split('/').pop()?.replace(/\.soul$/, '') || ''
    if (!name) return
    try {
      await trainingJobsController.loadCheckpoint(name)
      addToast(`Loaded trained version: ${name}`, 'success')
    } catch {
      addToast('Failed to load trained version', 'error')
    }
  }, [status?.checkpoint, addToast])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Knowledge Distillation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Teach a compact student model from a larger teacher. The student learns to match the teacher&apos;s output distribution.
        </p>

        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs">Teacher model</Label>
            <Input
              size={1}
              value={teacherModel}
              onChange={e => setTeacherModel(e.target.value)}
              placeholder="gpt2"
              className="text-sm"
            />
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">Dataset</Label>
            <Select value={selectedDataset} onValueChange={setSelectedDataset}>
              <SelectTrigger className="text-sm">
                <SelectValue placeholder="Select a dataset..." />
              </SelectTrigger>
              <SelectContent>
                {ds.datasets.map(d => (
                  <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">Temperature: {temperature.toFixed(1)}</Label>
            <Slider value={[temperature]} onValueChange={(v) => setTemperature(v[0])} min={1} max={10} step={0.5} />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Epochs</Label>
              <Input size={1} type="number" value={epochs} onChange={e => setEpochs(Number(e.target.value))} min={1} max={200} className="text-sm" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Embed dim</Label>
              <Input size={1} type="number" value={embedDim} onChange={e => setEmbedDim(Number(e.target.value))} min={16} max={512} step={16} className="text-sm" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Layers</Label>
              <Input size={1} type="number" value={nLayers} onChange={e => setNLayers(Number(e.target.value))} min={1} max={12} className="text-sm" />
            </div>
          </div>
        </div>

        {running && status && (
          <div className="space-y-2" role="status" aria-live="polite">
            <p className="text-sm text-muted-foreground">
              Distilling... epoch {status.current_epoch || '?'}/{epochs}
              {status.train_loss != null && ` loss: ${status.train_loss.toFixed(4)}`}
            </p>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${status.progress}%` }} />
            </div>
            {status.loss_history && status.loss_history.length > 1 && (
              <div className="h-16 w-full">
                <svg viewBox="0 0 100 16" className="w-full h-full">
                  <polyline
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="0.5"
                    className="text-primary"
                    points={status.loss_history.map((p, i) =>
                      `${(i / Math.max(status.loss_history!.length - 1, 1)) * 100},${Math.max(0, Math.min(15, 15 - p.value * 3))}`
                    ).join(' ')}
                  />
                </svg>
              </div>
            )}
          </div>
        )}

        {!running && status?.status === 'completed' && (
          <div className="rounded-lg border border-success/20 bg-success/5 p-3 space-y-2">
            <p className="text-sm font-medium text-success">Distillation complete</p>
            {status.train_loss != null && <p className="text-xs text-muted-foreground">Final loss: {status.train_loss.toFixed(4)}</p>}
            <div className="flex gap-2 mt-1">
              <Button size="sm" onClick={handleLoadCheckpoint}>Load for chat</Button>
              <Button size="sm" variant="outline" onClick={() => { setStatus(null); setJobId(null) }}>Dismiss</Button>
            </div>
          </div>
        )}

        {status?.status === 'failed' && (
          <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3">
            <p className="text-sm font-medium text-destructive">Distillation failed</p>
            <p className="text-xs text-muted-foreground">{status.error}</p>
            <Button size="sm" variant="outline" onClick={() => { setStatus(null); setJobId(null) }} className="mt-1">Dismiss</Button>
          </div>
        )}

        <Button size="sm" disabled={running || !selectedDataset} onClick={startDistill}>
          {running ? 'Distilling...' : 'Start distillation'}
        </Button>
      </CardContent>
    </Card>
  )
}
