'use client'

import { useState, memo, useEffect, useCallback } from 'react'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  Input,
  Label,
  Progress,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { StatTileGrid, type StatTile } from '@/components/training/stat-tile'
import { formatDuration } from '@/lib/formatDuration'
import { trainingJobsController } from '@/lib/training-controller'
import { experimentsController, type Experiment } from '@/lib/experiments-controller'
import { formatToastError } from '@/lib/error-utils'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'

export const TURBO_DEFAULTS = { epochs: 10, lr: 1e-3, embed: 128, heads: 4, layers: 3 }

export type QuickTrainConfig = {
  epochs: number
  lr: number
  embed: number
  heads: number
  layers: number
}

export const QuickTrainCard = memo(function QuickTrainCard({
  datasets,
  session,
  addToast,
}: {
  datasets: UseTrainingDatasetsReturn
  session: UseTrainingSessionReturn
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}) {
  const [config, setConfig] = useState<QuickTrainConfig>(TURBO_DEFAULTS)
  const [loadingModel, setLoadingModel] = useState(false)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>('')
  const running = session.turboPhase === 'training'
  const turboPaused = session.paused
  const turboProgressKnown =
    session.turboProgress > 0 || session.turboGlobalStep > 0 || session.turboTotalSteps > 0
  const turboProgressValue: number | null = turboProgressKnown ? session.turboProgress : null

  const pauseTraining = useCallback(async () => {
    try {
      await trainingJobsController.pauseTraining()
    } catch (e) {
      addToast(formatToastError(e, 'Could not pause training'), 'error')
    }
  }, [addToast])

  const resumeTraining = useCallback(async () => {
    try {
      await trainingJobsController.resumeTraining()
    } catch (e) {
      addToast(formatToastError(e, 'Could not resume training'), 'error')
    }
  }, [addToast])

  useEffect(() => {
    let active = true
    experimentsController
      .list()
      .then((data) => {
        if (active) setExperiments(data)
      })
      .catch((e) => {
        if (active) addToast(formatToastError(e, 'Could not load experiments'), 'error')
      })
    return () => {
      active = false
    }
  }, [addToast])

  const [starting, setStarting] = useState(false)

  const start = async () => {
    if (!datasets.selectedDataset) {
      addToast('Select a dataset first', 'error')
      return
    }
    setStarting(true)
    try {
      await session.startTurboTrain(
        datasets.selectedDataset,
        config,
        addToast,
        selectedExperimentId || undefined,
      )
    } catch (e) {
      addToast(formatToastError(e, 'Could not start turbo training'), 'error')
    } finally {
      setStarting(false)
    }
  }

  const loadForChat = async () => {
    const path = session.turboResult?.model_path ?? session.turboResult?.checkpoint
    const name = path ? path.split('/').pop() : null
    if (!name) {
      addToast('No model path to load', 'error')
      return
    }
    setLoadingModel(true)
    try {
      await trainingJobsController.loadCheckpoint(name)
      addToast(`Loaded trained version: ${name}`, 'success')
    } catch (e) {
      addToast(formatToastError(e, 'Could not load trained version'), 'error')
    } finally {
      setLoadingModel(false)
    }
  }

  const RANGES: Record<keyof QuickTrainConfig, [number, number]> = {
    epochs: [1, 500],
    lr: [1e-6, 1],
    embed: [16, 1024],
    heads: [1, 32],
    layers: [1, 64],
  }
  const setNum = (key: keyof QuickTrainConfig) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = Number(e.target.value)
    if (Number.isNaN(raw) || raw <= 0) return
    const [lo, hi] = RANGES[key]
    const clamped = Math.max(lo, Math.min(hi, Math.round(raw)))
    setConfig((prev) => ({ ...prev, [key]: clamped }))
  }

  // Conditional live stats as data — absent values simply don't produce a tile,
  // so the row fills without the grid-cols-4 hole the old markup left at 6 items.
  const runningTiles: StatTile[] = []
  if (session.turboTotalSteps > 0)
    runningTiles.push({
      key: 'step',
      label: 'Step',
      value: session.turboGlobalStep,
      hint: `/${session.turboTotalSteps}`,
    })
  if (session.turboLoss != null)
    runningTiles.push({ key: 'loss', label: 'Loss', value: session.turboLoss.toFixed(4) })
  if (session.turboStepsPerSec != null)
    runningTiles.push({
      key: 'speed',
      label: 'Speed',
      value: session.turboStepsPerSec.toFixed(1),
      hint: ' steps/s',
    })
  if (session.turboEta != null)
    runningTiles.push({ key: 'eta', label: 'ETA', value: formatDuration(session.turboEta) })
  runningTiles.push({
    key: 'elapsed',
    label: 'Elapsed',
    value: formatDuration(session.turboElapsedSeconds),
  })
  if (session.avgQuality != null)
    runningTiles.push({
      key: 'quality',
      label: 'Quality',
      value: session.avgQuality.toFixed(1),
      hint: '/5',
    })

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Training — advanced options</CardTitle>
        <p className="text-xs text-muted-foreground font-mono">
          epochs · lr · embed_dim · n_heads · n_layers · experiment_id
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {running ? (
          <div className="space-y-3" aria-live="polite" aria-atomic="true">
            <Progress value={turboProgressValue} max={100} />
            <StatTileGrid tiles={runningTiles} />
            <div className="flex items-center gap-2 border-t border-border/60 pt-3">
              <Button variant="destructive" size="sm" onClick={session.stopTurboTrain}>
                Stop
              </Button>
              {turboPaused ? (
                <Button variant="outline" size="sm" onClick={resumeTraining}>
                  Resume
                </Button>
              ) : (
                <Button variant="outline" size="sm" onClick={pauseTraining}>
                  Pause
                </Button>
              )}
            </div>
          </div>
        ) : session.turboPhase === 'complete' ? (
          <div className="space-y-2 text-sm">
            <p className="text-success">Training complete!</p>
            {session.turboResult && (
              <div className="space-y-1 rounded-lg bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                <p>
                  Final loss:{' '}
                  {typeof session.turboResult.final_loss === 'number'
                    ? session.turboResult.final_loss.toFixed(4)
                    : '--'}
                </p>
                <p>Steps: {session.turboResult.total_steps ?? '--'}</p>
                {session.avgQuality != null && <p>Quality: {session.avgQuality.toFixed(1)}/5</p>}
                {session.turboResult.model_path && (
                  <p className="truncate">Model: {session.turboResult.model_path}</p>
                )}
              </div>
            )}
            <div className="flex items-center gap-2 border-t border-border/60 pt-3">
              <Button size="sm" onClick={loadForChat} disabled={loadingModel}>
                {loadingModel ? 'Loading…' : 'Load for chat'}
              </Button>
              <Button variant="outline" size="sm" onClick={session.stopTurboTrain}>
                Train another
              </Button>
            </div>
          </div>
        ) : session.turboPhase === 'error' ? (
          <StatusBanner
            variant="error"
            message={session.turboError || 'Training failed'}
            dismissible={false}
            onDismiss={session.stopTurboTrain}
          />
        ) : (
          <div className="space-y-4">
            <div className="rounded-lg bg-muted/30 p-3">
              <DatasetSelector
                datasets={datasets}
                value={datasets.selectedDataset}
                onChange={datasets.setSelectedDataset}
                showImport
              />
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-epochs" variant="uppercase">
                  Epochs
                </Label>
                <Input
                  id="turbo-epochs"
                  type="number"
                  min={1}
                  max={500}
                  value={config.epochs}
                  onChange={setNum('epochs')}
                  className="h-8 text-xs font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-lr" variant="uppercase">
                  LR
                </Label>
                <Input
                  id="turbo-lr"
                  type="text"
                  inputMode="decimal"
                  value={config.lr}
                  onChange={setNum('lr')}
                  className="h-8 text-xs font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-embed" variant="uppercase">
                  Embed
                </Label>
                <Input
                  id="turbo-embed"
                  type="number"
                  min={16}
                  max={1024}
                  value={config.embed}
                  onChange={setNum('embed')}
                  className="h-8 text-xs font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-heads" variant="uppercase">
                  Heads
                </Label>
                <Input
                  id="turbo-heads"
                  type="number"
                  min={1}
                  max={32}
                  value={config.heads}
                  onChange={setNum('heads')}
                  className="h-8 text-xs font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-layers" variant="uppercase">
                  Layers
                </Label>
                <Input
                  id="turbo-layers"
                  type="number"
                  min={1}
                  max={64}
                  value={config.layers}
                  onChange={setNum('layers')}
                  className="h-8 text-xs font-mono"
                />
              </div>
            </div>
            {experiments.length > 0 && (
              <div className="flex flex-col gap-1">
                <Label htmlFor="turbo-experiment" variant="uppercase">
                  Experiment
                </Label>
                <Select value={selectedExperimentId} onValueChange={setSelectedExperimentId}>
                  <SelectTrigger
                    id="turbo-experiment"
                    className="h-8 text-xs font-mono"
                    aria-label="Experiment"
                  >
                    <SelectValue placeholder="None" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">None</SelectItem>
                    {experiments.map((exp) => (
                      <SelectItem key={exp.id} value={exp.id}>
                        {exp.name || exp.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            <div className="flex items-center gap-2 border-t border-border/60 pt-3">
              <Button
                size="sm"
                onClick={start}
                disabled={!datasets.selectedDataset || starting}
                className="font-mono"
              >
                {starting ? 'Starting...' : 'Start turbo train'}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
})
