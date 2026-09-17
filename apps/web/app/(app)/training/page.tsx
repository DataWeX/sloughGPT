'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { PageContainer } from '@/components/PageContainer'
import { Button } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { datasetController } from '@/lib/controllers'
import { trainingJobsController } from '@/lib/training-controller'
import { useTrainingForm } from '@/hooks/useTrainingForm'
import { TestModelDialog } from '@/components/training/TestModelDialog'
import { useApiReady } from '@/hooks/useLiveStatus'
import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { useTestDialog } from '@/hooks/useTestDialog'
import { TrainingPipeline } from '@/components/training/TrainingPipeline'
import { QuickTrainCard } from '@/components/training/QuickTrainCard'
import { StopTrainingButton } from '@/components/training/StopTrainingButton'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

export default function TrainingPage() {
  const searchParams = useSearchParams()
  const addToast = useToastStore((s) => s.addToast)
  const initialLoadDone = useRef(false)
  const session = useTrainingSession()
  const datasets = useTrainingDatasets(addToast)
  const checkpoints = useTrainingCheckpoints()
  const test = useTestDialog()
  const [pipelineStep, setPipelineStep] = useState<'data' | 'configure' | 'train' | 'results'>(
    'data',
  )
  const [completedSteps, setCompletedSteps] = useState<
    Set<'data' | 'configure' | 'train' | 'results'>
  >(new Set())

  const form = useTrainingForm(datasets, session, checkpoints, addToast)
  useRefreshShortcut(() => {
    void datasets.fetchDatasets()
    void checkpoints.fetchCheckpoints()
    void checkpoints.fetchJobs()
  })

  // Pause checkpoint polling when page is hidden
  const visibilityRef = useRef<boolean>(true)
  const ready = useApiReady()
  const tickRef = useRef<() => void>(() => {})
  tickRef.current = () => {
    if (visibilityRef.current) {
      void checkpoints.fetchCheckpoints()
      const hasRunning = form.allJobs.some((j) => j.status === 'running')
      if (hasRunning) void checkpoints.fetchJobs()
    }
  }
  useEffect(() => {
    if (!ready) return
    const onVisibility = () => {
      visibilityRef.current = !document.hidden
    }
    document.addEventListener('visibilitychange', onVisibility)
    const id = setInterval(() => tickRef.current(), 10000)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [ready])

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

  useEffect(() => {
    let active = true
    if (datasets.selectedDataset && form.inputMode === 'dataset') {
      datasetController
        .preview(datasets.selectedDataset, 3)
        .then((preview) => {
          if (active) datasets.setDatasetPreview(preview)
        })
        .catch(() => {
          if (active) datasets.setDatasetPreview(null)
        })
    } else {
      datasets.setDatasetPreview(null)
    }
    return () => {
      active = false
    }
  }, [datasets.selectedDataset, form.inputMode])

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

  const requestNotificationPermission = () => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }
  useEffect(() => {
    requestNotificationPermission()
  }, [])

  const runningJob = form.allJobs.find((j) => j.status === 'running')
  const completedCount = form.allJobs.filter((j) => j.status === 'completed').length

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
      {/* 3-step pipeline */}
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
        onStepComplete={(id) => setCompletedSteps((prev) => new Set(prev).add(id))}
      />

      {/* Quick train alternative */}
      <QuickTrainCard datasets={datasets} session={session} addToast={addToast} />

      <TestModelDialog
        open={test.testDialogOpen}
        prompt={test.testPrompt}
        result={test.testResult}
        loading={test.testLoading}
        streaming={test.testStreaming}
        streamingText={test.testStreamingText}
        responseFormat={test.responseFormat}
        onClose={() => test.setTestDialogOpen(false)}
        onPromptChange={test.setTestPrompt}
        onGenerate={test.handleTestModel}
        onClear={test.clearTest}
        onResponseFormatChange={test.setResponseFormat}
      />
    </PageContainer>
  )
}
