'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress, cn } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { trainingJobsController } from '@/lib/training-controller'
import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { LossChart, type LossPoint } from '@/components/training/LossChart'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { formatDuration } from '@/components/training/formatDuration'
import { TrainingLogCard } from '@/components/training/TrainingLogCard'
import { StopTrainingButton } from '@/components/training/StopTrainingButton'
import { useApiReady } from '@/hooks/useLiveStatus'

type InputMode = 'text' | 'dataset' | 'checkpoint'

export default function AutoTrainPage() {
  const addToast = useToastStore(s => s.addToast)
  const session = useTrainingSession()
  const datasets = useTrainingDatasets(addToast)
  const checkpoints = useTrainingCheckpoints()
  const ready = useApiReady()

  const [inputMode, setInputMode] = useState<InputMode>('text')
  const [sourceText, setSourceText] = useState('')
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('')
  const [epochs, setEpochs] = useState(5)
  const [learningRate, setLearningRate] = useState(1e-3)
  const [teacherModel, setTeacherModel] = useState('gpt2')
  const [temperature, setTemperature] = useState(1.0)

  const trainingRunning = session.trainingRunning

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
      epochs,
      learning_rate: learningRate,
      teacher_model: teacherModel,
      temperature,
    }

    if (inputMode === 'text') {
      body.source_text = sourceText
    } else if (inputMode === 'dataset') {
      body.dataset_id = datasets.selectedDataset
    } else if (inputMode === 'checkpoint') {
      body.checkpoint_name = selectedCheckpoint
    }

    session.startSSETraining(body, addToast, () => {
      void checkpoints.fetchCheckpoints()
    })
  }, [canStart, inputMode, sourceText, datasets.selectedDataset, selectedCheckpoint, epochs, learningRate, teacherModel, temperature, session, addToast, checkpoints])

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
      await trainingJobsController.stopAutoTrain()
      session.stopTraining()
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

  const loadCheckpoint = useCallback(async (name: string) => {
    try {
      await trainingJobsController.loadCheckpoint(name)
      addToast(`Loaded checkpoint: ${name}`, 'success')
    } catch {
      addToast('Could not load checkpoint', 'error')
    }
  }, [addToast])

  const deleteCheckpoint = useCallback(async (name: string) => {
    try {
      await trainingJobsController.deleteCheckpoint(name)
      addToast(`Deleted checkpoint: ${name}`, 'success')
      void checkpoints.fetchCheckpoints()
    } catch {
      addToast('Could not delete checkpoint', 'error')
    }
  }, [addToast, checkpoints])

  const [selectedCps, setSelectedCps] = useState<Set<string>>(new Set())
  const [cpPage, setCpPage] = useState(0)
  const CP_PAGE_SIZE = 10

  const toggleCpSelect = useCallback((name: string) => {
    setSelectedCps(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name); else next.add(name)
      return next
    })
  }, [])

  const batchDeleteCheckpoints = useCallback(async () => {
    if (selectedCps.size === 0) return
    try {
      const names = Array.from(selectedCps)
      await trainingJobsController.deleteCheckpointsBatch(names)
      addToast(`Deleted ${names.length} checkpoints`, 'success')
      setSelectedCps(new Set())
      void checkpoints.fetchCheckpoints()
    } catch {
      addToast('Batch delete failed', 'error')
    }
  }, [selectedCps, addToast, checkpoints])

  const downloadCheckpoint = useCallback(async (name: string) => {
    try {
      const blob = await trainingJobsController.downloadCheckpoint(name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = name; a.click()
      URL.revokeObjectURL(url)
    } catch {
      addToast('Download failed', 'error')
    }
  }, [addToast])

  const [checkpointInfo, setCheckpointInfo] = useState<Record<string, unknown> | null>(null)
  const [checkpointInfoName, setCheckpointInfoName] = useState('')

  const fetchCheckpointInfo = useCallback(async (name: string) => {
    setCheckpointInfoName(name)
    try {
      const info = await trainingJobsController.getCheckpointInfo(name)
      setCheckpointInfo(info)
    } catch {
      addToast('Could not load checkpoint info', 'error')
      setCheckpointInfo(null)
    }
  }, [addToast])

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
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Checkpoints</p>
            <p className="text-base font-medium">{checkpoints.checkpoints.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Trained</p>
            <p className="text-base font-medium">{completedCount}</p>
          </CardContent>
        </Card>
        <Card role="status" aria-live="polite">
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Status</p>
            <p className="text-base font-medium">
              {trainingRunning ? 'Training' : 'Idle'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Current loss</p>
            <p className="text-base font-medium font-mono">
              {session.loss != null ? session.loss.toFixed(4) : '--'}
            </p>
          </CardContent>
        </Card>
      </div>

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

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-epochs" variant="uppercase">Epochs</Label>
                  <Input id="at-epochs" type="number" min={1} max={500} value={epochs}
                    onChange={e => setEpochs(Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-lr" variant="uppercase">LR</Label>
                  <Input id="at-lr" type="text" inputMode="decimal" value={learningRate}
                    onChange={e => setLearningRate(Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-teacher" variant="uppercase">Teacher</Label>
                  <Input id="at-teacher" type="text" value={teacherModel}
                    onChange={e => setTeacherModel(e.target.value)} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="at-temp" variant="uppercase">Temperature</Label>
                  <Input id="at-temp" type="text" inputMode="decimal" value={temperature}
                    onChange={e => setTemperature(Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
              </div>

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

      {checkpoints.checkpoints.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Checkpoints ({checkpoints.checkpoints.length})</CardTitle>
              <div className="flex gap-1">
                {selectedCps.size > 0 && (
                  <Button size="sm" variant="ghost" className="text-destructive text-xs" onClick={() => void batchDeleteCheckpoints()}>
                    Delete {selectedCps.size}
                  </Button>
                )}
                <Button size="sm" variant="ghost" className="text-xs" onClick={() => {
                  if (selectedCps.size === checkpoints.checkpoints.length) setSelectedCps(new Set())
                  else setSelectedCps(new Set(checkpoints.checkpoints.map(c => c.name)))
                }}>
                  {selectedCps.size === checkpoints.checkpoints.length ? 'Deselect all' : 'Select all'}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {checkpoints.checkpoints.slice(cpPage * CP_PAGE_SIZE, (cpPage + 1) * CP_PAGE_SIZE).map(c => (
                <div key={c.name} className={cn('flex items-center justify-between rounded border p-3 text-sm', selectedCps.has(c.name) && 'border-primary bg-primary/5')}>
                  <div className="min-w-0 flex-1 flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={selectedCps.has(c.name)}
                      onChange={() => toggleCpSelect(c.name)}
                      aria-label={`Select checkpoint ${c.name}`}
                      className="h-3.5 w-3.5 rounded border-border"
                    />
                    <div className="min-w-0">
                      <p className="truncate font-medium">{c.name}</p>
                    </div>
                    <div className="flex gap-3 text-xs text-muted-foreground">
                      {c.loss != null && <span>Loss {c.loss.toFixed(4)}</span>}
                      {c.steps != null && <span>{c.steps} steps</span>}
                      {c.epochs != null && <span>{c.epochs} epochs</span>}
                      {c.size_mb != null && <span>{c.size_mb.toFixed(1)} MB</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="sm" variant="ghost" onClick={() => void fetchCheckpointInfo(c.name)}>Info</Button>
                    <Button size="sm" variant="ghost" onClick={() => void downloadCheckpoint(c.name)}>Download</Button>
                    <Button size="sm" variant="ghost" onClick={() => loadCheckpoint(c.name)}>Load</Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => deleteCheckpoint(c.name)}>Delete</Button>
                  </div>
                </div>
              ))}
            </div>
            {checkpoints.checkpoints.length > CP_PAGE_SIZE && (
              <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/30">
                <span className="text-xs text-muted-foreground">
                  {cpPage * CP_PAGE_SIZE + 1}–{Math.min((cpPage + 1) * CP_PAGE_SIZE, checkpoints.checkpoints.length)} of {checkpoints.checkpoints.length}
                </span>
                <div className="flex gap-1">
                  <Button size="sm" variant="ghost" className="text-xs" disabled={cpPage === 0} onClick={() => setCpPage(p => p - 1)}>Prev</Button>
                  <Button size="sm" variant="ghost" className="text-xs" disabled={(cpPage + 1) * CP_PAGE_SIZE >= checkpoints.checkpoints.length} onClick={() => setCpPage(p => p + 1)}>Next</Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {checkpointInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setCheckpointInfo(null)}>
          <Card className="w-full max-w-lg max-h-[80vh] overflow-auto" onClick={e => e.stopPropagation()}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Checkpoint: {checkpointInfoName}</CardTitle>
                <Button size="sm" variant="ghost" onClick={() => setCheckpointInfo(null)}>Close</Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-1.5 text-xs">
                {Object.entries(checkpointInfo).map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-border/30 py-1">
                    <span className="text-muted-foreground">{k}</span>
                    <span className="font-mono text-right">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </PageContainer>
  )
}
