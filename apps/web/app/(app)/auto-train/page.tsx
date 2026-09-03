'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'

import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { LossChart, type LossPoint } from '@/components/training/LossChart'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { formatDuration } from '@/components/training/formatDuration'
import { TrainingLogCard } from '@/components/training/TrainingLogCard'
import { StopTrainingButton } from '@/components/training/StopTrainingButton'
import { CheckpointManager } from '@/components/training/CheckpointManager'
import { AutoTrainKpiGrid } from '@/components/training/AutoTrainKpiGrid'
import { useApiReady } from '@/hooks/useLiveStatus'

type InputMode = 'text' | 'dataset' | 'checkpoint'

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(Math.max(value, min), max)
}

function sanitizeText(value: string, maxLen: number = 200): string {
  return value
    .replace(/[<>"'&]/g, '')
    .slice(0, maxLen)
}

function parseNumericInput(raw: string, min: number, max: number, fallback: number): number {
  const n = Number(raw)
  return Number.isFinite(n) ? clamp(n, min, max) : fallback
}

interface Preset {
  name: string
  description: string
  values: Partial<TrainingConfig>
}

interface TrainingConfig {
  epochs: number
  learning_rate: number
  batch_size: number
  teacher_model: string
  temperature: number
  soul_name: string
  early_stopping_patience: number
  n_embed: number
  n_layer: number
  n_head: number
  block_size: number
  dropout: number
}

const PRESETS: Preset[] = [
  {
    name: 'Quick test',
    description: 'Fast iteration with minimal compute',
    values: { epochs: 3, learning_rate: 1e-3, batch_size: 8, early_stopping_patience: 0 },
  },
  {
    name: 'Balanced',
    description: 'Good quality with reasonable speed',
    values: { epochs: 20, learning_rate: 3e-4, batch_size: 16, early_stopping_patience: 5 },
  },
  {
    name: 'High quality',
    description: 'Best results, slower training',
    values: { epochs: 100, learning_rate: 1e-4, batch_size: 32, early_stopping_patience: 10 },
  },
  {
    name: 'Low resource',
    description: 'Small model for limited hardware',
    values: {
      epochs: 10, learning_rate: 5e-4, batch_size: 4,
      n_embed: 64, n_layer: 2, n_head: 2, block_size: 64, dropout: 0.2,
    },
  },
  {
    name: 'Large model',
    description: 'Bigger capacity for complex patterns',
    values: {
      epochs: 50, learning_rate: 2e-4, batch_size: 8,
      n_embed: 256, n_layer: 8, n_head: 8, block_size: 256, dropout: 0.1,
    },
  },
]

const DEFAULT_CONFIG: TrainingConfig = {
  epochs: 20,
  learning_rate: 3e-4,
  batch_size: 16,
  teacher_model: 'gpt2',
  temperature: 0.8,
  soul_name: 'assistant',
  early_stopping_patience: 5,
  n_embed: 128,
  n_layer: 4,
  n_head: 4,
  block_size: 128,
  dropout: 0.1,
}

export default function AutoTrainPage() {
  const addToast = useToastStore(s => s.addToast)
  const session = useTrainingSession()
  const datasets = useTrainingDatasets(addToast)
  const checkpoints = useTrainingCheckpoints()
  const ready = useApiReady()

  const [inputMode, setInputMode] = useState<InputMode>('text')
  const [sourceText, setSourceText] = useState('')
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('')
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const trainingRunning = session.trainingRunning

  const updateConfig = useCallback(<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
    let safe: TrainingConfig[K] = value
    switch (key) {
      case 'epochs': safe = clamp(Number(value), 1, 1000) as TrainingConfig[K]; break
      case 'learning_rate': safe = clamp(Number(value), 1e-5, 1.0) as TrainingConfig[K]; break
      case 'batch_size': safe = clamp(Number(value), 1, 1024) as TrainingConfig[K]; break
      case 'temperature': safe = clamp(Number(value), 0.1, 2.0) as TrainingConfig[K]; break
      case 'early_stopping_patience': safe = clamp(Number(value), 0, 100) as TrainingConfig[K]; break
      case 'n_embed': safe = clamp(Number(value), 16, 1024) as TrainingConfig[K]; break
      case 'n_layer': safe = clamp(Number(value), 1, 24) as TrainingConfig[K]; break
      case 'n_head': safe = clamp(Number(value), 1, 64) as TrainingConfig[K]; break
      case 'block_size': safe = clamp(Number(value), 8, 2048) as TrainingConfig[K]; break
      case 'dropout': safe = clamp(Number(value), 0.0, 0.9) as TrainingConfig[K]; break
      case 'soul_name': safe = sanitizeText(String(value), 200) as TrainingConfig[K]; break
      case 'teacher_model': safe = sanitizeText(String(value), 100) as TrainingConfig[K]; break
    }
    setConfig(prev => ({ ...prev, [key]: safe }))
    setSelectedPreset(null)
  }, [])

  const applyPreset = useCallback((preset: Preset) => {
    setConfig(prev => ({ ...prev, ...preset.values }))
    setSelectedPreset(preset.name)
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      if (active) {
        await datasets.fetchDatasets()
        await checkpoints.fetchCheckpoints()
      }
    }
    void load()
    return () => { active = false }
  }, [])

  const canStart = !trainingRunning && (
    (inputMode === 'text' && sourceText.trim().length > 0) ||
    (inputMode === 'dataset' && datasets.selectedDataset) ||
    (inputMode === 'checkpoint' && selectedCheckpoint)
  )

  const startTraining = useCallback(async () => {
    if (!canStart) return

    const body: Record<string, unknown> = {
      epochs: config.epochs,
      learning_rate: config.learning_rate,
      batch_size: config.batch_size,
      teacher_model: sanitizeText(config.teacher_model, 100),
      temperature: config.temperature,
      soul_name: sanitizeText(config.soul_name, 200),
      early_stopping_patience: config.early_stopping_patience,
      n_embed: config.n_embed,
      n_layer: config.n_layer,
      n_head: config.n_head,
      block_size: config.block_size,
      dropout: config.dropout,
    }

    if (inputMode === 'text') {
      body.source_text = sourceText
        .replace(/<script[\s\S]*?<\/script>/gi, '')
        .replace(/<[^>]+>/g, '')
        .trim()
    } else if (inputMode === 'dataset') {
      body.dataset_id = datasets.selectedDataset
    } else if (inputMode === 'checkpoint') {
      body.checkpoint_name = selectedCheckpoint
    }

    session.startSSETraining(body, addToast, () => {
      void checkpoints.fetchCheckpoints()
    })
  }, [canStart, inputMode, sourceText, datasets.selectedDataset, selectedCheckpoint, config, session, addToast, checkpoints])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); void startTraining() }
      if (e.key === 'Escape' && trainingRunning) { e.preventDefault(); session.stopTraining() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [canStart, trainingRunning, session, startTraining])

  const stopTraining = useCallback(async () => {
    try {
      await session.stopTraining()
      addToast('Training stopped', 'success')
    } catch {
      addToast('Could not stop training', 'error')
    }
  }, [session, addToast])

  const pauseTraining = useCallback(async () => {
    try {
      await session.pauseTraining(addToast)
    } catch {
      addToast('Could not pause training', 'error')
    }
  }, [session, addToast])

  const resumeTraining = useCallback(async () => {
    try {
      await session.resumeTraining(addToast)
    } catch {
      addToast('Could not resume training', 'error')
    }
  }, [session, addToast])

  const tickRef = useRef<() => void>(() => {})
  tickRef.current = () => {
    if (!document.hidden) {
      void checkpoints.fetchCheckpoints()
    }
  }
  useEffect(() => {
    if (!ready) return
    const id = setInterval(() => tickRef.current(), 10000)
    return () => clearInterval(id)
  }, [ready])

  const chartData: LossPoint[] = useMemo(() => (session.lossHistory ?? []).map((p: { step: number; loss: number }) => ({
    step: p.step,
    value: p.loss,
    type: 'train' as const,
  })), [session.lossHistory])

  const completedCount = checkpoints.checkpoints.filter(c => c.final_train_loss != null).length

  return (
    <PageContainer
      title="Auto-train"
      subtitle="Teach your agent from data with knowledge distillation"
      headerRight={
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={() => { void checkpoints.fetchCheckpoints() }}>
            Refresh
          </Button>
        </div>
      }
    >
      <AutoTrainKpiGrid
        checkpointCount={checkpoints.checkpoints.length}
        completedCount={completedCount}
        trainingRunning={trainingRunning}
        loss={session.loss}
        loading={false}
      />

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Training pipeline</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {trainingRunning ? (
            <div className="space-y-3" aria-live="polite" aria-atomic="true">
              <Progress value={session.progress} max={100} />
              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-4">
                <span>Step {session.globalStep}/{session.totalSteps || '--'}</span>
                <span>{session.stepsPerSec != null ? `${session.stepsPerSec.toFixed(1)} steps/s` : '-- steps/s'}</span>
                <span>Epoch {session.epoch}/{session.totalEpochs || '--'}</span>
                <span>ETA {formatDuration(session.eta)}</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-4">
                <span>Elapsed {formatDuration(session.elapsedSeconds)}</span>
                {session.message && <span className="col-span-3 truncate">{session.message}</span>}
              </div>
              <div className="flex items-center gap-2">
                <StopTrainingButton onStop={stopTraining} addToast={addToast} />
                {session.paused ? (
                  <Button variant="outline" size="sm" onClick={resumeTraining}>Resume</Button>
                ) : (
                  <Button variant="outline" size="sm" onClick={pauseTraining}>Pause</Button>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Presets */}
              <div className="space-y-2">
                <Label variant="uppercase" className="text-xs">Presets</Label>
                <div className="flex flex-wrap gap-2">
                  {PRESETS.map(preset => (
                    <Button
                      key={preset.name}
                      size="sm"
                      variant={selectedPreset === preset.name ? 'default' : 'outline'}
                      onClick={() => applyPreset(preset)}
                      title={preset.description}
                    >
                      {preset.name}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Input mode tabs */}
              <div className="flex gap-2">
                {([
                  { mode: 'text' as InputMode, label: 'Text input' },
                  { mode: 'dataset' as InputMode, label: 'Dataset' },
                  { mode: 'checkpoint' as InputMode, label: 'Resume checkpoint' },
                ]).map(({ mode, label }) => (
                  <Button
                    key={mode}
                    size="sm"
                    variant={inputMode === mode ? 'default' : 'outline'}
                    onClick={() => setInputMode(mode)}
                  >
                    {label}
                  </Button>
                ))}
              </div>

              {inputMode === 'text' && (
                <div className="space-y-2">
                  <Label htmlFor="source-text">Source text</Label>
                  <textarea
                    id="source-text"
                    value={sourceText}
                    onChange={e => setSourceText(e.target.value)}
                    placeholder="Paste or type your training data here..."
                    rows={6}
                    aria-label="Source text for training"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-sm"
                  />
                  <p className="text-xs text-muted-foreground">
                    {sourceText.split('\n').filter(l => l.trim()).length} lines
                  </p>
                </div>
              )}

              {inputMode === 'dataset' && (
                <DatasetSelector
                  datasets={datasets}
                  value={datasets.selectedDataset}
                  onChange={datasets.setSelectedDataset}
                />
              )}

              {inputMode === 'checkpoint' && (
                <div className="space-y-2">
                  <Label htmlFor="at-checkpoint">Select checkpoint</Label>
                  <select
                    id="at-checkpoint"
                    value={selectedCheckpoint}
                    onChange={e => setSelectedCheckpoint(e.target.value)}
                    aria-label="Select checkpoint"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="">-- select --</option>
                    {checkpoints.checkpoints.map(c => (
                      <option key={c.name} value={c.name}>
                        {c.name} {c.loss != null ? `(${c.loss.toFixed(4)})` : ''}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Core parameters */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-soul" variant="uppercase">Soul name</Label>
                  <Input id="at-soul" type="text" value={config.soul_name}
                    onChange={e => updateConfig('soul_name', e.target.value)} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-epochs" variant="uppercase">Epochs</Label>
                  <Input id="at-epochs" type="number" min={1} max={1000} value={config.epochs}
                    onChange={e => updateConfig('epochs', Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-lr" variant="uppercase">LR</Label>
                  <Input id="at-lr" type="number" min={1e-5} max={1.0} step={1e-4} value={config.learning_rate}
                    onChange={e => updateConfig('learning_rate', Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-batch" variant="uppercase">Batch size</Label>
                  <Input id="at-batch" type="number" min={1} max={1024} value={config.batch_size}
                    onChange={e => updateConfig('batch_size', Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-teacher" variant="uppercase">Teacher</Label>
                  <Input id="at-teacher" type="text" value={config.teacher_model}
                    onChange={e => updateConfig('teacher_model', e.target.value)} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-temp" variant="uppercase">Temperature</Label>
                  <Input id="at-temp" type="number" min={0.1} max={2.0} step={0.1} value={config.temperature}
                    onChange={e => updateConfig('temperature', Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-early-stop" variant="uppercase">Early stop</Label>
                  <Input id="at-early-stop" type="number" min={0} max={100} value={config.early_stopping_patience}
                    onChange={e => updateConfig('early_stopping_patience', Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div />
              </div>

              {/* Advanced architecture params */}
              <div>
                <Button
                  variant="ghost" size="sm"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-xs text-muted-foreground"
                >
                  {showAdvanced ? 'Hide' : 'Show'} advanced settings
                </Button>
              </div>

              {showAdvanced && (
                <div className="space-y-3 rounded-md border border-border p-3">
                  <p className="text-xs text-muted-foreground">Native training architecture parameters</p>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                    <div className="flex flex-col gap-1">
                      <Label htmlFor="at-n-embed" variant="uppercase">Embed dim</Label>
                      <Input id="at-n-embed" type="number" min={16} max={1024} value={config.n_embed}
                        onChange={e => updateConfig('n_embed', Number(e.target.value))} className="h-8 text-xs font-mono" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <Label htmlFor="at-n-layer" variant="uppercase">Layers</Label>
                      <Input id="at-n-layer" type="number" min={1} max={24} value={config.n_layer}
                        onChange={e => updateConfig('n_layer', Number(e.target.value))} className="h-8 text-xs font-mono" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <Label htmlFor="at-n-head" variant="uppercase">Heads</Label>
                      <Input id="at-n-head" type="number" min={1} max={64} value={config.n_head}
                        onChange={e => updateConfig('n_head', Number(e.target.value))} className="h-8 text-xs font-mono" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <Label htmlFor="at-block" variant="uppercase">Block size</Label>
                      <Input id="at-block" type="number" min={8} max={2048} value={config.block_size}
                        onChange={e => updateConfig('block_size', Number(e.target.value))} className="h-8 text-xs font-mono" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <Label htmlFor="at-dropout" variant="uppercase">Dropout</Label>
                      <Input id="at-dropout" type="number" min={0.0} max={0.9} step={0.05} value={config.dropout}
                        onChange={e => updateConfig('dropout', Number(e.target.value))} className="h-8 text-xs font-mono" />
                    </div>
                  </div>
                </div>
              )}

              <Button size="sm" onClick={startTraining} disabled={!canStart}>
                Start auto-train
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {chartData.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Loss curve</CardTitle>
          </CardHeader>
          <CardContent>
            <LossChart data={chartData} height={240} live={trainingRunning} />
          </CardContent>
        </Card>
      )}

      <TrainingLogCard trainingRunning={trainingRunning} />

      <CheckpointManager checkpoints={checkpoints.checkpoints} addToast={addToast} onRefresh={() => void checkpoints.fetchCheckpoints()} />
    </PageContainer>
  )
}
