'use client'

import { useState, memo, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress } from '@sloughgpt/strui'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { formatDuration } from '@/components/training/formatDuration'
import { trainingJobsController } from '@/lib/training-controller'
import { experimentsController, type Experiment } from '@/lib/experiments-controller'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'

export const TURBO_DEFAULTS = { epochs: 10, lr: 1e-3, embed: 128, heads: 4, layers: 3 }

export type TurboConfig = { epochs: number; lr: number; embed: number; heads: number; layers: number }

export const TurboCard = memo(function TurboCard({
  datasets,
  session,
  addToast,
}: {
  datasets: UseTrainingDatasetsReturn
  session: UseTrainingSessionReturn
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}) {
  const [config, setConfig] = useState<TurboConfig>(TURBO_DEFAULTS)
  const [loadingModel, setLoadingModel] = useState(false)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>('')
  const running = session.turboPhase === 'training'
  const turboPaused = session.paused

  const pauseTraining = useCallback(async () => {
    try {
      await trainingJobsController.pauseTraining()
    } catch {
      addToast('Could not pause training', 'error')
    }
  }, [addToast])

  const resumeTraining = useCallback(async () => {
    try {
      await trainingJobsController.resumeTraining()
    } catch {
      addToast('Could not resume training', 'error')
    }
  }, [addToast])

  useEffect(() => {
    let active = true
    experimentsController.list().then(data => { if (active) setExperiments(data) }).catch(() => { if (active) addToast('Could not load experiments', 'error') })
    return () => { active = false }
  }, [addToast])

  const start = () => {
    if (!datasets.selectedDataset) {
      addToast('Select a dataset first', 'error')
      return
    }
    session.startTurboTrain(datasets.selectedDataset, config, addToast, selectedExperimentId || undefined)
  }

  const loadForChat = async () => {
    const path = session.turboResult?.model_path
    const name = path ? path.split('/').pop() : null
    if (!name) {
      addToast('No model path to load', 'error')
      return
    }
    setLoadingModel(true)
    try {
      await trainingJobsController.loadCheckpoint(name)
      addToast(`Loaded trained version: ${name}`, 'success')
    } catch {
      addToast('Could not load trained version', 'error')
    } finally {
      setLoadingModel(false)
    }
  }

  const setNum = (key: keyof TurboConfig) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setConfig(prev => ({ ...prev, [key]: Number(e.target.value) }))

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Fast train (turbo)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {running ? (
          <div className="space-y-3" aria-live="polite" aria-atomic="true">
            <Progress value={session.turboProgress} max={100} />
            <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-4">
              <span>
                Step {session.turboGlobalStep}/{session.turboTotalSteps || '--'}
              </span>
              <span>{session.turboStepsPerSec != null ? `${session.turboStepsPerSec.toFixed(1)} steps/s` : '-- steps/s'}</span>
              <span>ETA {formatDuration(session.turboEta)}</span>
              <span>Elapsed {formatDuration(session.turboElapsedSeconds)}</span>
            </div>
            {session.turboLoss != null && <p className="text-xs text-muted-foreground">Loss {session.turboLoss.toFixed(4)}</p>}
            {session.avgQuality != null && <p className="text-xs text-muted-foreground">Quality {session.avgQuality.toFixed(1)}/5</p>}
            <div className="flex items-center gap-2">
              <Button variant="destructive" size="sm" onClick={session.stopTurboTrain}>
                Stop
              </Button>
              {turboPaused ? (
                <Button variant="outline" size="sm" onClick={resumeTraining}>Resume</Button>
              ) : (
                <Button variant="outline" size="sm" onClick={pauseTraining}>Pause</Button>
              )}
            </div>
          </div>
        ) : session.turboPhase === 'complete' ? (
          <div className="space-y-2 text-sm">
            <p className="text-success">Turbo training complete!</p>
            {session.turboResult && (
              <div className="space-y-1 text-xs text-muted-foreground">
                <p>
                  Final loss: {typeof session.turboResult.final_loss === 'number' ? session.turboResult.final_loss.toFixed(4) : '--'}
                </p>
                <p>Steps: {session.turboResult.total_steps ?? '--'}</p>
                {session.avgQuality != null && <p>Quality: {session.avgQuality.toFixed(1)}/5</p>}
                {session.turboResult.model_path && <p className="truncate">Model: {session.turboResult.model_path}</p>}
              </div>
            )}
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={loadForChat} disabled={loadingModel}>
                {loadingModel ? 'Loading…' : 'Load for chat'}
              </Button>
              <Button variant="outline" size="sm" onClick={session.stopTurboTrain}>
                Train another
              </Button>
            </div>
          </div>
        ) : session.turboPhase === 'error' ? (
          <div className="space-y-2 text-sm">
            <p className="text-destructive">Turbo training failed</p>
            {session.turboError && <p className="text-xs text-muted-foreground">{session.turboError}</p>}
            <Button variant="outline" size="sm" onClick={session.stopTurboTrain}>
              Dismiss
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <DatasetSelector
              datasets={datasets}
              value={datasets.selectedDataset}
              onChange={datasets.setSelectedDataset}
              showImport
            />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-epochs" variant="uppercase">Epochs</Label>
                <Input id="turbo-epochs" type="number" min={1} max={500} value={config.epochs}
                  onChange={setNum('epochs')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-lr" variant="uppercase">LR</Label>
                <Input id="turbo-lr" type="text" inputMode="decimal" value={config.lr}
                  onChange={setNum('lr')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-embed" variant="uppercase">Embed</Label>
                <Input id="turbo-embed" type="number" min={16} max={1024} value={config.embed}
                  onChange={setNum('embed')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-heads" variant="uppercase">Heads</Label>
                <Input id="turbo-heads" type="number" min={1} max={32} value={config.heads}
                  onChange={setNum('heads')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-layers" variant="uppercase">Layers</Label>
                <Input id="turbo-layers" type="number" min={1} max={64} value={config.layers}
                  onChange={setNum('layers')} className="h-8 text-xs font-mono" />
              </div>
            </div>
            {experiments.length > 0 && (
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-experiment" variant="uppercase">Experiment</Label>
                <select id="turbo-experiment" value={selectedExperimentId}
                  onChange={e => setSelectedExperimentId(e.target.value)}
                  className="h-8 text-xs font-mono rounded-md border border-border bg-background px-2">
                  <option value="">None</option>
                  {experiments.map(exp => (
                    <option key={exp.id} value={exp.id}>{exp.name || exp.id}</option>
                  ))}
                </select>
              </div>
            )}
            <Button size="sm" onClick={start} disabled={!datasets.selectedDataset}>
              Start turbo train
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
})
