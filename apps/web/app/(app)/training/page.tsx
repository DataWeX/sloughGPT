'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useCallback, useRef, useState, useMemo } from 'react'
import { useSearchParams } from 'next/navigation'
import { PageContainer } from '@/components/PageContainer'
import { Button, Skeleton, cn } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { writeTraining } from '@/lib/app-shell'
import { datasetController } from '@/lib/controllers'
import { trainingJobsController } from '@/lib/training-controller'
import { useTrainingForm } from '@/hooks/useTrainingForm'
import { TestModelDialog } from '@/components/training/TestModelDialog'
import { useApiReady } from '@/hooks/useLiveStatus'
import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { useTestDialog } from '@/hooks/useTestDialog'
import { TrainingSummaryCard } from '@/components/training/TrainingSummaryCard'
import { TrainingHealthCard } from '@/components/training/TrainingHealthCard'
import { TrainingPipeline } from '@/components/training/TrainingPipeline'
import { QuickTrainCard } from '@/components/training/QuickTrainCard'
import { APILogsCard } from '@/components/training/APILogsCard'
import { FeedbackTrainCard } from '@/components/training/FeedbackTrainCard'
import { RecoveryCard } from '@/components/training/RecoveryCard'
import { WebhooksCard } from '@/components/training/WebhooksCard'
import { TrainingBuildsCard } from '@/components/training/TrainingBuildsCard'
import { TrainingDataCard } from '@/components/training/TrainingDataCard'
import { SessionTrainingCard } from '@/components/training/SessionTrainingCard'
import { TrainingLogCard } from '@/components/training/TrainingLogCard'
import { TrainingLiveChart } from '@/components/training/TrainingLiveChart'
import { TrainingHistoryView } from '@/components/training/TrainingHistoryView'
import { TrainingAnalyticsCard } from '@/components/training/TrainingAnalyticsCard'
import { StopTrainingButton } from '@/components/training/StopTrainingButton'

type ManualTab = 'train' | 'results' | 'settings'

const MANUAL_TABS: { id: ManualTab; label: string }[] = [
  { id: 'train', label: 'Train' },
  { id: 'results', label: 'Results' },
  { id: 'settings', label: 'Settings' },
]

export default function TrainingPage() {
  const searchParams = useSearchParams()
  const addToast = useToastStore(s => s.addToast)
  const initialLoadDone = useRef(false)
  const session = useTrainingSession()
  const datasets = useTrainingDatasets(addToast)
  const checkpoints = useTrainingCheckpoints()
  const test = useTestDialog()
  const [manualTab, setManualTab] = useState<ManualTab>('train')
  const [pipelineStep, setPipelineStep] = useState<'data' | 'configure' | 'train' | 'results'>('data')
  const [completedSteps, setCompletedSteps] = useState<Set<'data' | 'configure' | 'train' | 'results'>>(new Set())

  const form = useTrainingForm(datasets, session, checkpoints, addToast)

  // ===== Visibility-based polling pause =====
  const visibilityRef = useRef<boolean>(true)

  // Warn before leaving during active training
  useEffect(() => {
    if (!session.trainingRunning) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [session.trainingRunning])

  // Keyboard shortcuts for training page
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        if (form.canStart && !session.trainingRunning) {
          form.startTraining()
          addToast('Training started', 'success')
        }
      }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
        e.preventDefault()
        test.setTestDialogOpen(true)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [form.canStart, session.trainingRunning, form.startTraining, addToast, test.setTestDialogOpen])

  useEffect(() => {
    void datasets.fetchDatasets()
    void checkpoints.fetchCheckpoints()
    void checkpoints.fetchJobs()
    const urlDataset = searchParams.get('dataset')
    if (urlDataset) {
      datasets.setSelectedDataset(urlDataset)
    } else if (!initialLoadDone.current && datasets.datasets.length > 0) {
      initialLoadDone.current = true
      datasets.setSelectedDataset(datasets.datasets[0].id)
    }
  }, [searchParams])

  // Pause checkpoint polling when page is hidden
  const ready = useApiReady()
  const tickRef = useRef<() => void>(() => {})
  tickRef.current = () => {
    if (visibilityRef.current) {
      void checkpoints.fetchCheckpoints()
      const hasRunning = form.allJobs.some(j => j.status === 'running')
      if (hasRunning) void checkpoints.fetchJobs()
    }
  }
  useEffect(() => {
    if (!ready) return
    const onVisibility = () => { visibilityRef.current = !document.hidden }
    document.addEventListener('visibilitychange', onVisibility)
    const id = setInterval(() => tickRef.current(), 10000)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [ready])

  useEffect(() => {
    let active = true
    if (datasets.selectedDataset && form.inputMode === 'dataset') {
      datasetController.preview(datasets.selectedDataset, 3).then(preview => {
        if (active) datasets.setDatasetPreview(preview)
      }).catch(() => { if (active) datasets.setDatasetPreview(null) })
    } else {
      datasets.setDatasetPreview(null)
    }
    return () => { active = false }
  }, [datasets.selectedDataset, form.inputMode])

  const runningJob = useMemo(() => form.allJobs.find(j => j.status === 'running'), [form.allJobs])
  const completedCount = useMemo(() => form.allJobs.filter(j => j.status === 'completed').length, [form.allJobs])

  // If checkpoints polling finds a running job but session isn't tracking it, sync the state
  useEffect(() => {
    if (runningJob && !session.trainingRunning) {
      writeTraining({
        phase: 'TRAINING',
        method: (runningJob.method as 'slonet' | 'hf' | 'turbo' | null) || 'slonet',
        loss: runningJob.loss ?? runningJob.train_loss ?? null,
        progress: runningJob.progress ?? 0,
        epoch: runningJob.current_epoch ?? 0,
        totalEpochs: runningJob.epochs ?? 0,
        globalStep: runningJob.global_step ?? 0,
        totalSteps: runningJob.total_steps ?? 0,
        stepsPerSec: runningJob.steps_per_sec ?? null,
        eta: runningJob.eta_s ?? null,
        elapsedSeconds: runningJob.elapsed_s ?? null,
        jobId: runningJob.id,
      })
    }
  }, [runningJob, session.trainingRunning])

  // Browser notification on training completion
  const prevJobStatusesRef = useRef<Map<string, string>>(new Map())
  useEffect(() => {
    const prev = prevJobStatusesRef.current
    for (const job of form.allJobs) {
      const prevStatus = prev.get(job.id)
      if (prevStatus === 'running' && job.status === 'completed') {
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('Training Complete', {
            body: `${job.name || 'Training job'} finished successfully`,
            icon: '/favicon.svg',
          })
        }
      }
    }
    const next = new Map<string, string>()
    for (const job of form.allJobs) {
      next.set(job.id, job.status)
    }
    prevJobStatusesRef.current = next
  }, [form.allJobs])

  const requestNotificationPermission = useCallback(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [])

  useEffect(() => {
    requestNotificationPermission()
  }, [requestNotificationPermission])

  const handleExportMetrics = useCallback(async () => {
    try {
      const blob = await trainingJobsController.exportMetrics()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'training-metrics.json'
      a.click()
      URL.revokeObjectURL(url)
      addToast('Metrics exported', 'success')
    } catch {
      addToast('Could not export metrics', 'error')
    }
  }, [addToast])

  const handlePurgeJobs = useCallback(async () => {
    try {
      const result = await trainingJobsController.purgeJobs()
      addToast(`Purged ${result.purged} completed jobs`, 'success')
      void checkpoints.fetchJobs()
    } catch {
      addToast('Could not purge jobs', 'error')
    }
  }, [addToast, checkpoints])

  return (
    <PageContainer
      title="Teach me"
      subtitle="Teach your agent from your data"
      className="items-start"
      loading={checkpoints.loadingJobs && form.allJobs.length === 0}
      headerRight={
        <div className="flex items-center gap-2">
          {runningJob && (
            <StopTrainingButton
              onStop={async () => {
                await trainingJobsController.stop(runningJob.id)
                void checkpoints.fetchJobs()
              }}
              addToast={addToast}
            />
          )}
          <Button size="sm" variant="ghost" onClick={handleExportMetrics}>Export metrics</Button>
          {completedCount > 0 && (
            <Button size="sm" variant="ghost" onClick={handlePurgeJobs}>Purge completed</Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              void checkpoints.fetchJobs()
              void checkpoints.fetchCheckpoints()
            }}
          >
            Refresh
          </Button>
        </div>
      }
    >
        {/* Stats */}
        <KpiGrid columns={4}>
          <StatCard label="Training runs" value={checkpoints.loadingJobs ? <Skeleton className="h-5 w-8 inline-block" /> : form.allJobs.length} />
          <StatCard label="Running" value={checkpoints.loadingJobs ? <Skeleton className="h-5 w-8 inline-block" /> : (runningJob ? 1 : 0)} />
          <StatCard label="Completed" value={checkpoints.loadingJobs ? <Skeleton className="h-5 w-8 inline-block" /> : completedCount} />
          <StatCard label="Saved versions" value={checkpoints.loadingCheckpoints ? <Skeleton className="h-5 w-8 inline-block" /> : checkpoints.checkpoints.length} />
        </KpiGrid>

        <TrainingSummaryCard checkpoints={checkpoints.checkpoints} loading={checkpoints.loadingCheckpoints} />

        <TrainingHealthCard checkpoints={checkpoints.checkpoints} loading={checkpoints.loadingCheckpoints} />

        {session.trainingRunning && (
          <TrainingLiveChart
            lossHistory={session.lossHistory}
            progress={session.progress}
            epoch={session.epoch}
            totalEpochs={session.totalEpochs}
            globalStep={session.globalStep}
            totalSteps={session.totalSteps}
            stepsPerSec={session.stepsPerSec}
            eta={session.eta}
          />
        )}

        <TrainingLogCard trainingRunning={session.trainingRunning} />

        {/* Manual sub-tabs */}
        <div className="border-b border-border/50" role="tablist" aria-label="Training sections">
          <div className="flex gap-0">
            {MANUAL_TABS.map(t => (
              <button
                type="button"
                role="tab"
                id={`tab-${t.id}`}
                key={t.id}
                aria-selected={manualTab === t.id}
                aria-controls={`panel-${t.id}`}
                onClick={() => setManualTab(t.id)}
                className={cn(
                  'relative px-4 py-2.5 text-xs font-medium transition-colors',
                  manualTab === t.id
                    ? 'text-primary'
                    : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {t.label}
                {manualTab === t.id && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-full" />
                )}
              </button>
            ))}
          </div>
        </div>

            {/* Train tab */}
            {manualTab === 'train' && (
              <div role="tabpanel" id="panel-train" aria-labelledby="tab-train">
                {/* Pipeline */}
                <TrainingPipeline
                  form={form}
                  datasets={datasets}
                  session={session}
                  checkpoints={checkpoints}
                  onTest={() => test.setTestDialogOpen(true)}
                  addToast={addToast}
                  step={pipelineStep}
                  onStepChange={setPipelineStep}
                  completedSteps={completedSteps}
                  onStepComplete={(id) => setCompletedSteps(prev => new Set(prev).add(id))}
                />

                {/* Fast train (turbo) */}
                <QuickTrainCard datasets={datasets} session={session} addToast={addToast} />

                {/* Train from feedback */}
                <FeedbackTrainCard addToast={addToast} />

                {/* Train from API conversation logs */}
                <APILogsCard addToast={addToast} />
              </div>
            )}

            {/* Results tab */}
            {manualTab === 'results' && (
              <div role="tabpanel" id="panel-results" aria-labelledby="tab-results">
                {/* Training analytics */}
                <TrainingAnalyticsCard addToast={addToast} />

                {/* Training history */}
                <TrainingHistoryView addToast={addToast} />

                {/* All builds */}
                <TrainingBuildsCard addToast={addToast} />

                {/* Training data */}
                <TrainingDataCard addToast={addToast} />

                {/* Session training */}
                <SessionTrainingCard addToast={addToast} />
              </div>
            )}

            {/* Settings tab */}
            {manualTab === 'settings' && (
              <div role="tabpanel" id="panel-settings" aria-labelledby="tab-settings">
                {/* Recoverable jobs */}
                <RecoveryCard addToast={addToast} />

                {/* Webhooks */}
                <WebhooksCard addToast={addToast} />
              </div>
            )}

      <TestModelDialog
        open={test.testDialogOpen}
        prompt={test.testPrompt}
        result={test.testResult}
        loading={test.testLoading}
        onClose={() => test.setTestDialogOpen(false)}
        onPromptChange={test.setTestPrompt}
        onGenerate={test.handleTestModel}
        onClear={test.clearTest}
      />
    </PageContainer>
  )
}
