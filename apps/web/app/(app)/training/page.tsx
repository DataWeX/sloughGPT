'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useCallback, useRef, useState, useMemo } from 'react'
import { useSearchParams } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { Tabs } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { datasetController, modelController } from '@/lib/controllers'
import { trainingJobsController } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'
import { useTrainingForm } from '@/hooks/useTrainingForm'
import { OutputCard } from '@/components/OutputCard'
import { TestModelDialog } from '@/components/training/TestModelDialog'
import { useApiReady } from '@/hooks/useLiveStatus'

import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { useTestDialog } from '@/hooks/useTestDialog'
import { JobHistoryCard } from '@/components/training/JobHistoryCard'
import { CheckpointsCard } from '@/components/training/CheckpointsCard'
import { FineTunedModelsCard } from '@/components/training/FineTunedModelsCard'
import { TrainingFormCard } from '@/components/training/TrainingFormCard'
import { TrainFromSessionsCard } from '@/components/training/TrainFromSessionsCard'
import { TrainingDataCard } from '@/components/training/TrainingDataCard'
import { EvalReportCard } from '@/components/training/EvalReportCard'
import { SelfTrainCard } from '@/components/training/SelfTrainCard'
import { DatasetPreviewCard } from '@/components/training/DatasetPreviewCard'
import { CheckpointCompareCard } from '@/components/training/CheckpointCompareCard'
import { BestCheckpointCard } from '@/components/training/BestCheckpointCard'
import { TrainingSummaryCard } from '@/components/training/TrainingSummaryCard'
import { CheckpointLossChart } from '@/components/training/CheckpointLossChart'
import { CheckpointNotes } from '@/components/training/CheckpointNotes'
import { TrainingHealthCard } from '@/components/training/TrainingHealthCard'
import { useCheckpointFilter } from '@/components/training/useCheckpointFilter'
import { CheckpointFilterBar } from '@/components/training/CheckpointFilterBar'
import { useTrainingSearch, TrainingSearchBar } from '@/components/training/TrainingSearch'
import { TrainingRunCard } from '@/components/training/TrainingRunCard'
import { TrainingProgress } from '@/components/training/TrainingProgress'
import { TrainingTimeline } from '@/components/training/TrainingTimeline'
import { TrainingQuickActions } from '@/components/training/TrainingQuickActions'
import { TrainingCompareCard } from '@/components/training/TrainingCompareCard'
import { TrainingDashboard } from '@/components/training/TrainingDashboard'
import { TrainingOnboarding } from '@/components/training/TrainingOnboarding'
import { TrainingTips } from '@/components/training/TrainingTips'
import { TrainingActivity } from '@/components/training/TrainingActivity'

export default function TrainingPage() {
  const searchParams = useSearchParams()
  const addToast = useToastStore(s => s.addToast)
  const initialLoadDone = useRef(false)
  const session = useTrainingSession()
  const datasets = useTrainingDatasets(addToast)
  const checkpoints = useTrainingCheckpoints()
  const test = useTestDialog()

  const form = useTrainingForm(datasets, session, checkpoints, addToast)

  const [activeTab, setActiveTab] = useState('train')
  const [currentModelId, setCurrentModelId] = useState<string | null>(null)

  useEffect(() => {
    void modelController.status().then(s => setCurrentModelId(s.model_type))
  }, [])

  // ===== Visibility-based polling pause =====
  const visibilityRef = useRef<boolean>(true)

  // ===== Loading timeout: show retry suggestion if data doesn't load in 15s =====
  const [loadingTimedOut, setLoadingTimedOut] = useState(false)
  useEffect(() => {
    if (checkpoints.loadingCheckpoints || checkpoints.loadingJobs) {
      setLoadingTimedOut(false)
      const t = setTimeout(() => setLoadingTimedOut(true), 15000)
      return () => clearTimeout(t)
    }
  }, [checkpoints.loadingCheckpoints, checkpoints.loadingJobs])

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
      if (e.key === 't' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setActiveTab('train')
      }
      if (e.key === 'h' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setActiveTab('history')
      }
      if (e.key === 'e' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setActiveTab('eval')
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
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

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
      addToast('Failed to export metrics', 'error')
    }
  }, [addToast])

  const { filtered: filteredCheckpoints, typeFilter, setTypeFilter, lossMax, setLossMax, types } = useCheckpointFilter(checkpoints.checkpoints)
  const { filtered: searchResults, query, setQuery } = useTrainingSearch(filteredCheckpoints)

  const bestName = useMemo(() => {
    const valid = searchResults.filter(c => c.loss != null && c.loss > 0)
    if (!valid.length) return null
    return valid.reduce((a, b) => (a.loss ?? Infinity) < (b.loss ?? Infinity) ? a : b).name
  }, [searchResults])

  const TABS = [
    { value: 'train', label: 'Train' },
    { value: 'history', label: 'History', count: checkpoints.checkpoints.length || undefined },
  ]

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        className="items-start"
        left={<AppRouteHeaderLead title="Teach me" subtitle="Teach your agent from your data" />}
        right={
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" onClick={handleExportMetrics}>Export metrics</Button>
            <Button size="sm" variant="ghost" onClick={() => { void checkpoints.fetchJobs(); void checkpoints.fetchCheckpoints() }}>Refresh</Button>
          </div>
        }
      />

      <div className="space-y-4">
        {/* Stats */}
        <KpiGrid columns={4}>
          <StatCard label="Training runs" value={form.allJobs.length} />
          <StatCard label="Running" value={runningJob ? 1 : 0} />
          <StatCard label="Completed" value={completedCount} />
          <StatCard label="Saved versions" value={checkpoints.checkpoints.length} />
        </KpiGrid>

        <TrainingSummaryCard checkpoints={checkpoints.checkpoints} />

        <TrainingHealthCard checkpoints={checkpoints.checkpoints} />

        {/* Tabs */}
        <Tabs value={activeTab} onChange={setActiveTab} tabs={TABS} />

        {/* Tab: Train */}
        {activeTab === 'train' && (
          <div className="space-y-4">
            <TrainingFormCard
              form={form}
              datasets={datasets}
              session={session}
              checkpoints={checkpoints}
              onTest={() => test.setTestDialogOpen(true)}
            />

            {datasets.selectedDataset && (
              <DatasetPreviewCard datasetId={datasets.selectedDataset} />
            )}

            <TrainFromSessionsCard />

            <TrainingDataCard />

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Schedule</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full bg-success" />
                    <span>Training runs immediately when started</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full bg-primary" />
                    <span>Queue multiple jobs with &ldquo;Start training&rdquo;</span>
                  </div>
                </div>
                {form.allJobs.filter(j => j.status === 'running').length > 0 && (
                  <p className="text-[11px] text-warning mt-2">
                    {form.allJobs.filter(j => j.status === 'running').length} job(s) running — new jobs will queue automatically
                  </p>
                )}
              </CardContent>
            </Card>

            <SelfTrainCard />

            {!session.trainingRunning && !checkpoints.loadingCheckpoints && checkpoints.checkpoints.length === 0 && form.allJobs.length === 0 && (
              <Card className="border-dashed">
                <CardHeader>
                  <CardTitle className="text-base text-muted-foreground">No training activity yet</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-xs text-muted-foreground/70 space-y-1.5 max-w-sm text-left">
                    <p>Pick a dataset above and click <span className="font-medium text-foreground">Start training</span></p>
                    <p>Use <span className="font-medium text-foreground">Paste text</span> to train on your own content</p>
                    <p>Conversation-format data (JSONL) trains better than plain text</p>
                    <p>Small datasets (5-10 samples) work for personality fine-tuning</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Tab: History */}
        {activeTab === 'history' && (
          <div className="space-y-4">
            <TrainingOnboarding
              hasCheckpoints={checkpoints.checkpoints.length > 0}
              onStartTraining={() => setActiveTab('train')}
              onImportData={() => setActiveTab('train')}
            />

            <TrainingDashboard checkpoints={checkpoints.checkpoints} />

            <TrainingTips
              checkpoints={checkpoints.checkpoints}
              isTraining={!!runningJob}
              hasDataset={!!datasets.selectedDataset}
            />

            <TrainingActivity checkpoints={checkpoints.checkpoints} />

            <TrainingProgress job={runningJob ?? null} />

            <TrainingTimeline checkpoints={searchResults} />

            <TrainingQuickActions
              checkpoints={searchResults}
              onLoadBest={async (name) => {
                try {
                  await soulsController.loadCheckpoint(name)
                  addToast(`Loaded ${name}`, 'success')
                  await modelController.status()
                } catch {
                  addToast('Failed to load checkpoint', 'error')
                }
              }}
              onExportMetrics={handleExportMetrics}
            />

            <TrainingSearchBar query={query} onQueryChange={setQuery} total={filteredCheckpoints.length} shown={searchResults.length} />

            <CheckpointFilterBar
              types={types}
              typeFilter={typeFilter}
              onTypeFilterChange={setTypeFilter}
              lossMax={lossMax}
              onLossMaxChange={setLossMax}
              total={checkpoints.checkpoints.length}
              shown={filteredCheckpoints.length}
            />

            <CheckpointsCard checkpoints={checkpoints} loadingTimedOut={loadingTimedOut} onRetry={() => { setLoadingTimedOut(false); void checkpoints.fetchCheckpoints() }} onContinue={form.startTraining} onTest={() => test.setTestDialogOpen(true)} />

            <BestCheckpointCard
              checkpoints={searchResults}
              onLoad={async (name) => {
                try {
                  await soulsController.loadCheckpoint(name)
                  addToast(`Loaded ${name}`, 'success')
                  await modelController.status()
                } catch {
                  addToast('Failed to load checkpoint', 'error')
                }
              }}
            />

            <CheckpointLossChart checkpoints={searchResults} />

            <div className="space-y-2">
              {searchResults.map((c, i) => (
                <TrainingRunCard
                  key={c.name}
                  checkpoint={c}
                  index={i}
                  isBest={c.name === bestName}
                  onLoad={async (cp) => {
                    try {
                      await soulsController.loadCheckpoint(cp.name)
                      addToast(`Loaded ${cp.name}`, 'success')
                      await modelController.status()
                    } catch {
                      addToast('Failed to load checkpoint', 'error')
                    }
                  }}
                />
              ))}
            </div>

            <CheckpointCompareCard checkpoints={searchResults} />

            <TrainingCompareCard
              checkpoints={searchResults}
              onLoad={async (name) => {
                try {
                  await soulsController.loadCheckpoint(name)
                  addToast(`Loaded ${name}`, 'success')
                  await modelController.status()
                } catch {
                  addToast('Failed to load checkpoint', 'error')
                }
              }}
            />

            <CheckpointNotes checkpoints={searchResults} />

            <FineTunedModelsCard activeModelId={currentModelId} onLoaded={() => { void checkpoints.fetchCheckpoints(); void modelController.status().then(s => setCurrentModelId(s.model_type)) }} />

            <EvalReportCard />

            <JobHistoryCard allJobs={form.allJobs} checkpoints={checkpoints} loadingTimedOut={loadingTimedOut} onRetry={() => { setLoadingTimedOut(false); void checkpoints.fetchJobs() }} />

            <OutputCard />
          </div>
        )}
      </div>

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
    </div>
  )
}
