'use client'

import { useCallback, useRef, useEffect } from 'react'
import { trainingFacade } from '@/lib/training-facade'
import { readTraining, writeTraining, type TrainingToastFn } from '@/lib/app-shell'
import { createSSEStream, type SSEEnvelope, type SSEStream } from '@/lib/sse-client'
import { logger } from '@/lib/dev-log'
import { formatToastError } from '@/lib/error-utils'

const _log = logger.child('training-stream')

const MAX_LOSS_HISTORY = 200

/**
 * Manages the SSE stream for distill (auto-train) training.
 * Parses the standard envelope and writes progress to the app shell.
 *
 * Returns a start function and provides cleanup on unmount.
 */
export function useTrainingStream() {
  const streamRef = useRef<SSEStream | null>(null)
  // Separates "we called stop()" from a genuine drop, so onClose stays quiet
  // after an intentional closeStream() (complete / error / unmount).
  const stoppedRef = useRef(false)

  const closeStream = useCallback(() => {
    stoppedRef.current = true
    streamRef.current?.stop()
    streamRef.current = null
  }, [])

  const startSSETraining = useCallback(
    (body: Record<string, unknown>, addToast: TrainingToastFn, onCheckpointUpdate?: () => void) => {
      closeStream()
      trainingFacade.jobs
        .startAutoTrain(body)
        .then(() => {
          writeTraining({
            phase: 'TRAINING',
            method: 'slonet',
            loss: null,
            progress: 0,
            epoch: 0,
            totalEpochs: 0,
            globalStep: 0,
            totalSteps: 0,
            eta: null,
            stepsPerSec: null,
            elapsedSeconds: null,
            message: '',
            lossHistory: [],
            evalResult: null,
            startTime: Date.now(),
            error: null,
            checkpoint: null,
            finalLoss: null,
            modelPath: null,
            jobId: null,
            avgQuality: null,
            dataQuality: null,
          })

          const handleEvent = (e: unknown) => {
            try {
              // The transport hands us its envelope type; the SSE payload is still
              // untyped wire data, so the fields feeding typed state are narrowed
              // at the point of use rather than blanket-`any` here.
              const env = e as Partial<SSEEnvelope>
              if (env.stream !== 'auto-train') return

              const patch: Record<string, unknown> = {}

              if (env.phase) patch.phase = env.phase
              if (env.data?.progress != null) patch.progress = env.data.progress
              if (env.data?.global_step != null) patch.globalStep = env.data.global_step
              if (env.data?.total_steps != null) patch.totalSteps = env.data.total_steps
              if (env.data?.eta_s != null) patch.eta = env.data.eta_s
              if (env.data?.steps_per_sec != null) patch.stepsPerSec = env.data.steps_per_sec
              if (env.data?.elapsed_s != null) patch.elapsedSeconds = env.data.elapsed_s
              if (env.meta?.epoch != null) patch.epoch = env.meta.epoch
              if (env.meta?.total_epochs != null) patch.totalEpochs = env.meta.total_epochs
              if (env.message) patch.message = env.message
              if (env.data?.eval_report) patch.evalResult = env.data.eval_report
              if (env.data?.avg_quality != null) patch.avgQuality = env.data.avg_quality

              // lossHistory accumulation — must read current state then write once
              // to avoid stale reads when both loss and eval_loss arrive together.
              if (env.data?.loss != null || env.data?.eval_loss != null) {
                const current = readTraining()
                let hist = current.lossHistory
                if (env.data?.loss != null) {
                  const step = (hist[hist.length - 1]?.step ?? 0) + 1
                  hist = [...hist, { step, loss: env.data.loss as number }]
                  patch.loss = env.data.loss
                }
                if (env.data?.eval_loss != null) {
                  const step = (env.data?.step as number | undefined) ?? hist.length
                  hist = [...hist, { step, loss: env.data.eval_loss as number, isEval: true }]
                }
                patch.lossHistory =
                  hist.length > MAX_LOSS_HISTORY ? hist.slice(-MAX_LOSS_HISTORY) : hist
              }

              if (Object.keys(patch).length > 0) writeTraining(patch)

              if (env.status === 'complete') {
                closeStream()
                const current = readTraining()
                writeTraining({
                  phase: 'complete',
                  checkpoint: (env.data?.checkpoint as string | null) ?? null,
                  finalLoss: (env.data?.final_loss as number | null) ?? null,
                  modelPath: (env.data?.model_path as string | null) ?? current.modelPath,
                  avgQuality: (env.data?.avg_quality as number | null) ?? current.avgQuality,
                  dataQuality:
                    (env.data?.data_quality as typeof current.dataQuality) ?? current.dataQuality,
                })
                addToast('Training complete', 'success')
                onCheckpointUpdate?.()
              }
              if (env.status === 'error') {
                closeStream()
                const errorMsg = env.message || (env.data?.error as string) || 'Training failed'
                writeTraining({ phase: 'error', error: errorMsg })
                addToast(errorMsg, 'error')
              }
            } catch (err) {
              _log.error('SSE parse error', { exception: String(err) })
            }
          }

          // startSSETraining closes any previous stream first, which sets
          // stoppedRef — a freshly opened stream is live again, so reset it
          // before handing control to the transport.
          stoppedRef.current = false
          const stream = createSSEStream({
            url: '/training/stream',
            // Same policy the EventSource had: three strikes, then tell the UI.
            maxReconnects: 3,
            onEvent: (e) => handleEvent(e),
            onClose: () => {
              if (stoppedRef.current) return // intentional closeStream()
              writeTraining({ phase: 'error', error: 'Connection lost' })
              addToast('Connection lost during training', 'error')
            },
          })
          streamRef.current = stream
          stream.start()
        })
        .catch((e) => addToast(formatToastError(e, 'Could not start training'), 'error'))
    },
    [closeStream],
  )

  useEffect(() => () => closeStream(), [closeStream])

  return { startSSETraining, closeStream }
}
