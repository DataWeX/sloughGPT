'use client'

import { useCallback, useRef, useEffect } from 'react'
import { trainingJobsController } from '@/lib/controllers'
import { readTraining, writeTraining, type TrainingToastFn } from '@/lib/app-shell'
import { PUBLIC_API_URL } from '@/lib/config'
import { logger } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'

const _log = logger.child('training-stream')

const MAX_LOSS_HISTORY = 200

/**
 * Manages the SSE EventSource for distill (auto-train) training.
 * Parses the standard envelope and writes progress to the app shell.
 *
 * Returns a start function and provides cleanup on unmount.
 */
export function useTrainingStream() {
  const esRef = useRef<EventSource | null>(null)

  const closeStream = useCallback(() => {
    esRef.current?.close(); esRef.current = null
  }, [])

  const startSSETraining = useCallback((
    body: Record<string, unknown>,
    addToast: TrainingToastFn,
    onCheckpointUpdate?: () => void,
  ) => {
    closeStream()
    trainingJobsController.startAutoTrain(body).then(() => {
      writeTraining({
        phase: 'TRAINING', method: 'slnet',
        loss: null, progress: 0, epoch: 0, totalEpochs: 0,
        globalStep: 0, totalSteps: 0, eta: null, stepsPerSec: null,
        elapsedSeconds: null, message: '', lossHistory: [], evalResult: null,
        startTime: Date.now(), error: null,
        checkpoint: null, finalLoss: null, modelPath: null, jobId: null,
        avgQuality: null, dataQuality: null,
      })

      const es = new EventSource(`${PUBLIC_API_URL}/auto-train/stream`)
      esRef.current = es

      es.onmessage = (e) => {
        try {
          const env = JSON.parse(e.data)
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
              hist = [...hist, { step, loss: env.data.loss }]
              patch.loss = env.data.loss
            }
            if (env.data?.eval_loss != null) {
              const step = env.data?.step ?? hist.length
              hist = [...hist, { step, loss: env.data.eval_loss, isEval: true }]
            }
            patch.lossHistory = hist.length > MAX_LOSS_HISTORY ? hist.slice(-MAX_LOSS_HISTORY) : hist
          }

          if (Object.keys(patch).length > 0) writeTraining(patch)

          if (env.status === 'complete') {
            closeStream()
            const current = readTraining()
            writeTraining({
              phase: 'complete',
              checkpoint: env.data?.checkpoint ?? null,
              finalLoss: env.data?.final_loss ?? null,
              avgQuality: env.data?.avg_quality ?? current.avgQuality,
              dataQuality: env.data?.data_quality ?? current.dataQuality,
            })
            addToast('Training complete', 'success')
            onCheckpointUpdate?.()
          }
          if (env.status === 'error') {
            closeStream()
            writeTraining({ phase: 'error', error: 'Training failed' })
            addToast('Training failed', 'error')
          }
        } catch (err) { _log.error('SSE parse error', { exception: String(err) }) }
      }

      let esRetries = 0
      es.onerror = () => {
        if (es.readyState === EventSource.CLOSED || esRetries >= 3) {
          closeStream()
          writeTraining({ phase: 'error', error: 'Connection lost' })
          addToast('Connection lost during training', 'error')
        } else { esRetries++ }
      }
    }).catch((e: unknown) => addToast(extractErrorMessage(e, 'Failed to start training'), 'error'))
  }, [closeStream])

  useEffect(() => () => closeStream(), [closeStream])

  return { startSSETraining, closeStream }
}
