'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useCallback, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { PageContainer } from '@/components/PageContainer'
import { Button, Skeleton } from '@sloughgpt/strui'
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
import { TurboCard } from '@/components/training/TurboCard'
import { APILogsCard } from '@/components/training/APILogsCard'
import { FeedbackTrainCard } from '@/components/training/FeedbackTrainCard'
import { RecoveryCard } from '@/components/training/RecoveryCard'
import { WebhooksCard } from '@/components/training/WebhooksCard'
import { TrainingBuildsCard } from '@/components/training/TrainingBuildsCard'
import { TrainingDataCard } from '@/components/training/TrainingDataCard'
import { SessionTrainingCard } from '@/components/training/SessionTrainingCard'
import { TrainingLogCard } from '@/components/training/TrainingLogCard'
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
    if (datasets.selectedDataset && form.inputMode === 'dataset') {
      datasetController.preview(datasets.selectedDataset, 3).then(datasets.setDatasetPreview).catch(() => datasets.setDatasetPreview(null))
    } else {
      datasets.setDatasetPreview(null)
    }
  }, [datasets.selectedDataset, form.inputMode])

  const runningJob = form.allJobs.find(j => j.status === 'running')
  const completedCount = form.allJobs.filter(j => j.status === 'completed').length

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
        if (Notification.permission === 'granted') {
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

        <TrainingSummaryCard checkpoints={checkpoints.checkpoints} />

        <TrainingHealthCard checkpoints={checkpoints.checkpoints} />

        <TrainingLogCard trainingRunning={session.trainingRunning} />

        {/* Manual sub-tabs */}
        <div className="flex flex-wrap gap-1.5">
          {MANUAL_TABS.map(t => (
            <button
              type="button"
              role="tab"
              key={t.id}
              aria-selected={manualTab === t.id}
              aria-label={`${t.label} tab`}
              onClick={() => setManualTab(t.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                manualTab === t.id ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

            {/* Train tab */}
            {manualTab === 'train' && (
              <>
                {/* Pipeline */}
                <TrainingPipeline
                  form={form}
                  datasets={datasets}
                  session={session}
                  checkpoints={checkpoints}
                  onTest={() => test.setTestDialogOpen(true)}
                  addToast={addToast}
                />

                {/* Fast train (turbo) */}
                <TurboCard datasets={datasets} session={session} addToast={addToast} />

                {/* Train from feedback */}
                <FeedbackTrainCard addToast={addToast} />

                {/* Train from API conversation logs */}
                <APILogsCard addToast={addToast} />
              </>
            )}

            {/* Results tab */}
            {manualTab === 'results' && (
              <>
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
              </>
            )}

            {/* Settings tab */}
            {manualTab === 'settings' && (
              <>
                {/* Recoverable jobs */}
                <RecoveryCard addToast={addToast} />

                {/* Webhooks */}
                <WebhooksCard addToast={addToast} />
              </>
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
