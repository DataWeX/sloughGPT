'use client'

import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button, Progress } from '@sloughgpt/strui'
import { TrainingErrorBanner } from '@/components/training/TrainingStatus'
import dynamic from 'next/dynamic'
import { trainingJobsController } from '@/lib/controllers'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'
import { DataStep } from '@/components/training/DataStep'
import { ConfigureStep } from '@/components/training/ConfigureStep'
import { TrainStep } from '@/components/training/TrainStep'
import { ResultsStep } from '@/components/training/ResultsStep'
import { formatDuration } from '@/components/training/formatDuration'

const LossChart = dynamic(() => import('@/components/training/LossChart').then(m => m.LossChart), { ssr: false })

const STEPS = [
  { id: 'data', label: 'Data', description: 'Pick your training data' },
  { id: 'configure', label: 'Configure', description: 'Set training parameters' },
  { id: 'train', label: 'Train', description: 'Run training' },
  { id: 'results', label: 'Results', description: 'View checkpoints & eval' },
] as const

type StepId = typeof STEPS[number]['id']

function StepIndicator({ current, completed, onStepClick }: { current: StepId; completed: Set<StepId>; onStepClick: (id: StepId) => void }) {
  return (
    <div className="flex items-center gap-1" role="navigation" aria-label="Training steps">
      {STEPS.map((step, i) => {
        const isDone = completed.has(step.id)
        const isCurrent = step.id === current
        const clickable = isDone && !isCurrent
        const content = (
          <>
            <div className={`flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-medium transition-colors ${
              isCurrent ? 'bg-primary text-primary-foreground' :
              isDone ? 'bg-primary/15 text-primary' :
              'bg-muted text-muted-foreground'
            }`}>
              {isDone ? '✓' : i + 1}
            </div>
            <span className={`text-xs ${isCurrent ? 'font-medium text-foreground' : 'text-muted-foreground'}`}>
              {step.label}
            </span>
          </>
        )
        return (
          <div key={step.id} className="flex items-center gap-1">
            {i > 0 && <div className={`w-6 h-px ${isDone || isCurrent ? 'bg-primary' : 'bg-border'}`} />}
            {clickable ? (
              <button
                type="button"
                onClick={() => onStepClick(step.id)}
                className="flex items-center gap-1.5 rounded-md transition-colors hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                aria-label={`Go to ${step.label} step`}
              >
                {content}
              </button>
            ) : (
              <div className="flex items-center gap-1.5">{content}</div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function TrainingPipeline({
  form,
  datasets,
  session,
  checkpoints,
  onTest,
  addToast,
}: {
  form: TrainingFormState
  datasets: UseTrainingDatasetsReturn
  session: UseTrainingSessionReturn
  checkpoints: UseTrainingCheckpointsReturn
  onTest: () => void
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}) {
  const [step, setStep] = useState<StepId>('data')
  const [completedSteps, setCompletedSteps] = useState<Set<StepId>>(new Set())

  const completeStep = (id: StepId) => setCompletedSteps(prev => new Set(prev).add(id))

  const runningJob = form.allJobs.find(j => j.status === 'running')
  const isTraining = session.trainingRunning || !!runningJob

  const stepIdx = STEPS.findIndex(s => s.id === step)

  const advance = () => {
    completeStep(step)
    const next = STEPS[stepIdx + 1]
    if (next) setStep(next.id)
  }

  const goBack = () => {
    const prev = STEPS[stepIdx - 1]
    if (prev) setStep(prev.id)
  }

  const goToTrain = () => setStep('train')

  const stepProps = { form, datasets, onNext: advance, onBack: goBack, addToast }

  if (isTraining) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Training in progress</CardTitle>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
              </span>
              {session.epoch > 0 && session.totalEpochs > 0 && (
                <span>Epoch {session.epoch}/{session.totalEpochs}</span>
              )}
              {session.loss != null && (
                <span>Loss: {session.loss.toFixed(4)}</span>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {session.lossHistory.length > 0 && (
            <LossChart data={session.lossHistory.map(p => ({ step: p.step, value: p.loss, type: 'train' as const }))} height={200} />
          )}

          {session.phase !== 'complete' && session.phase !== 'error' && (
            <div className="space-y-2">
              <Progress value={session.progress} max={100} label="Progress" showValue size="sm" />
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                {session.totalSteps > 0 && (
                  <span>Step {session.globalStep}/{session.totalSteps}</span>
                )}
                {session.stepsPerSec != null && session.stepsPerSec > 0 && (
                  <span>{session.stepsPerSec.toFixed(1)} steps/s</span>
                )}
                {session.eta != null && (
                  <span>ETA {formatDuration(session.eta)}</span>
                )}
                <span>Elapsed {formatDuration(session.elapsedSeconds)}</span>
              </div>
            </div>
          )}

          {session.phase === 'complete' && (
            <div className="space-y-3">
              <div className="rounded-md bg-success/10 border border-success/20 p-3 text-sm text-success">
                Training complete
                {session.distillCheckpoint && <span className="text-muted-foreground ml-1">— {session.distillCheckpoint}</span>}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" onClick={onTest}>Test model</Button>
                {session.distillCheckpoint && (
                  <Button size="sm" variant="outline" onClick={() => {
                    checkpoints.handleLoadCheckpoint(session.distillCheckpoint!, addToast)
                  }}>Load checkpoint</Button>
                )}
                {session.method === 'hf' && session.checkpoint && session.checkpoint.endsWith('.npz') && (
                  <Button size="sm" variant="outline" onClick={async () => {
                    try {
                      await trainingJobsController.loadAdapter(session.checkpoint!, false)
                      addToast('LoRA adapter loaded into model', 'success')
                    } catch (e) {
                      addToast('Failed to load adapter: ' + (e instanceof Error ? e.message : String(e)), 'error')
                    }
                  }}>Load LoRA adapter</Button>
                )}
                <Button size="sm" variant="ghost" onClick={() => {
                  session.resetTraining()
                  setStep('results')
                  completeStep('train')
                }}>View results</Button>
              </div>
            </div>
          )}

          {session.phase === 'error' && (
            <TrainingErrorBanner
              error={session.message || 'Training failed'}
              onRetry={session.resetTraining}
              onDismiss={session.resetTraining}
            />
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <StepIndicator current={step} completed={completedSteps} onStepClick={setStep} />

      {step === 'data' && <DataStep {...stepProps} />}
      {step === 'configure' && <ConfigureStep {...stepProps} />}
      {step === 'train' && <TrainStep {...stepProps} />}
      {step === 'results' && <ResultsStep checkpoints={checkpoints} goToTrain={goToTrain} onTest={onTest} addToast={addToast} />}
    </div>
  )
}
