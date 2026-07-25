'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { Tabs } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { datasetController } from '@/lib/controllers'
import { useTrainingForm } from '@/hooks/useTrainingForm'
import { OutputCard } from '@/components/OutputCard'
import { TestModelDialog } from '@/components/training/TestModelDialog'

import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { useTestDialog } from '@/hooks/useTestDialog'
import { JobHistoryCard } from '@/components/training/JobHistoryCard'
import { CheckpointsCard } from '@/components/training/CheckpointsCard'
import { TrainingFormCard } from '@/components/training/TrainingFormCard'
import { TrainFromSessionsCard } from '@/components/training/TrainFromSessionsCard'
import { TrainingDataCard } from '@/components/training/TrainingDataCard'
import { EvalReportCard } from '@/components/training/EvalReportCard'

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
  const tickRef = useRef<() => void>(() => {})
  tickRef.current = () => {
    if (visibilityRef.current) {
      void checkpoints.fetchCheckpoints()
      const hasRunning = form.allJobs.some(j => j.status === 'running')
      if (hasRunning) void checkpoints.fetchJobs()
    }
  }
  useEffect(() => {
    const onVisibility = () => { visibilityRef.current = !document.hidden }
    document.addEventListener('visibilitychange', onVisibility)
    const id = setInterval(() => tickRef.current(), 10000)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  useEffect(() => {
    if (datasets.selectedDataset && form.inputMode === 'dataset') {
      datasetController.preview(datasets.selectedDataset, 3).then(datasets.setDatasetPreview).catch(() => datasets.setDatasetPreview(null))
    } else {
      datasets.setDatasetPreview(null)
    }
  }, [datasets.selectedDataset, form.inputMode])

  const runningJob = form.allJobs.find(j => j.status === 'running')
  const completedCount = form.allJobs.filter(j => j.status === 'completed').length

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

            <TrainFromSessionsCard />

            <TrainingDataCard />

            {!session.trainingRunning && !checkpoints.loadingCheckpoints && checkpoints.checkpoints.length === 0 && form.allJobs.length === 0 && (
              <Card className="border-dashed py-8">
                <CardContent className="text-center space-y-3">
                  <p className="text-sm text-muted-foreground">No training activity yet.</p>
                  <div className="text-xs text-muted-foreground/70 space-y-1.5 max-w-sm mx-auto text-left">
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
            <CheckpointsCard checkpoints={checkpoints} loadingTimedOut={loadingTimedOut} onRetry={() => { setLoadingTimedOut(false); void checkpoints.fetchCheckpoints() }} onContinue={form.startTraining} onTest={() => test.setTestDialogOpen(true)} />

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
