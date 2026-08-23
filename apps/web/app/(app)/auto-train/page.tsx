'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { trainingJobsController } from '@/lib/training-controller'
import { datasetController } from '@/lib/controllers'
import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { LossChart, type LossPoint } from '@/components/training/LossChart'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { formatDuration } from '@/components/training/formatDuration'
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
  const [jobLogs, setJobLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [logsLoading, setLogsLoading] = useState(false)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const trainingRunning = session.trainingRunning

  useEffect(() => {
    void datasets.fetchDatasets()
    void checkpoints.fetchCheckpoints()
  }, [])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [jobLogs])

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

  const stopTraining = useCallback(async () => {
    try {
      await trainingJobsController.stopAutoTrain()
      session.stopTraining()
      addToast('Training stopped', 'success')
    } catch {
      addToast('Failed to stop training', 'error')
    }
  }, [session, addToast])

  const pauseTraining = useCallback(async () => {
    try {
      await session.pauseTraining(addToast)
    } catch {
      addToast('Failed to pause training', 'error')
    }
  }, [session, addToast])

  const resumeTraining = useCallback(async () => {
    try {
      await session.resumeTraining(addToast)
    } catch {
      addToast('Failed to resume training', 'error')
    }
  }, [session, addToast])

  const loadCheckpoint = useCallback(async (name: string) => {
    try {
      await trainingJobsController.loadCheckpoint(name)
      addToast(`Loaded checkpoint: ${name}`, 'success')
    } catch {
      addToast('Failed to load checkpoint', 'error')
    }
  }, [addToast])

  const deleteCheckpoint = useCallback(async (name: string) => {
    try {
      await trainingJobsController.deleteCheckpoint(name)
      addToast(`Deleted checkpoint: ${name}`, 'success')
      void checkpoints.fetchCheckpoints()
    } catch {
      addToast('Failed to delete checkpoint', 'error')
    }
  }, [addToast, checkpoints])

  const fetchLogs = useCallback(async () => {
    setLogsLoading(true)
    try {
      const logs = await trainingJobsController.getTrainingLog()
      setJobLogs(logs)
    } catch {
      addToast('Failed to fetch logs', 'error')
    } finally {
      setLogsLoading(false)
    }
  }, [addToast])

  const toggleLogs = useCallback(async () => {
    if (showLogs) {
      setShowLogs(false)
      return
    }
    await fetchLogs()
    setShowLogs(true)
  }, [showLogs, fetchLogs])

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

  const chartData: LossPoint[] = (session.lossHistory ?? []).map((p: { step: number; loss: number }) => ({
    step: p.step,
    value: p.loss,
    type: 'train' as const,
  }))

  const completedCount = checkpoints.checkpoints.filter(c => c.final_train_loss != null).length

  return (
    <PageContainer
      title="Auto-train"
      subtitle="Teach your agent from data with knowledge distillation"
      headerRight={
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={toggleLogs}>
            {showLogs ? 'Hide logs' : 'Show logs'}
          </Button>
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
            <p className="text-lg font-medium">{checkpoints.checkpoints.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Trained</p>
            <p className="text-lg font-medium">{completedCount}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Status</p>
            <p className="text-lg font-medium">
              {trainingRunning ? 'Training' : 'Idle'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Current loss</p>
            <p className="text-lg font-medium font-mono">
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
            <div className="space-y-3" aria-live="polite">
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
                <Button variant="destructive" size="sm" onClick={stopTraining}>Stop</Button>
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
                  <Label>Select checkpoint</Label>
                  <select
                    value={selectedCheckpoint}
                    onChange={e => setSelectedCheckpoint(e.target.value)}
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

      {showLogs && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Training logs</CardTitle>
          </CardHeader>
          <CardContent>
            {logsLoading ? (
              <p className="text-xs text-muted-foreground">Loading logs...</p>
            ) : jobLogs.length === 0 ? (
              <p className="text-xs text-muted-foreground">No logs yet.</p>
            ) : (
              <div className="max-h-64 overflow-y-auto rounded bg-muted/30 p-3 font-mono text-xs">
                {jobLogs.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
            <Button size="sm" variant="ghost" className="mt-2" onClick={fetchLogs} disabled={logsLoading}>
              Refresh logs
            </Button>
          </CardContent>
        </Card>
      )}

      {checkpoints.checkpoints.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Checkpoints ({checkpoints.checkpoints.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {checkpoints.checkpoints.map(c => (
                <div key={c.name} className="flex items-center justify-between rounded border p-3 text-sm">
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium">{c.name}</p>
                    <div className="flex gap-3 text-xs text-muted-foreground">
                      {c.loss != null && <span>Loss {c.loss.toFixed(4)}</span>}
                      {c.steps != null && <span>{c.steps} steps</span>}
                      {c.epochs != null && <span>{c.epochs} epochs</span>}
                      {c.size_mb != null && <span>{c.size_mb.toFixed(1)} MB</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="sm" variant="ghost" onClick={() => loadCheckpoint(c.name)}>Load</Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => deleteCheckpoint(c.name)}>Delete</Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
