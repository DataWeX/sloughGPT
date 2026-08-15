'use client'

import { useCallback, useRef, useEffect } from 'react'
import { trainingJobsController } from '@/lib/controllers'
import { readTraining, writeTraining, type TrainingToastFn } from '@/lib/app-shell'
import { PUBLIC_API_URL } from '@/lib/config'
import { logger } from '@/lib/dev-log'

const _log = logger.child('training-stream')

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
        phase: 'TRAINING', method: 'slnet', progress: 0, loss: null,
        epoch: 0, totalEpochs: 0, globalStep: 0, totalSteps: 0,
        eta: null, stepsPerSec: null, elapsedSeconds: null,
        message: '', lossHistory: [], evalResult: null,
        startTime: Date.now(), error: null,
      })

      const es = new EventSource(`${PUBLIC_API_URL}/auto-train/stream`)
      esRef.current = es

      es.onmessage = (e) => {
        try {
          const env = JSON.parse(e.data)
          if (env.stream !== 'auto-train') return

          writeTraining({ phase: env.phase || 'TRAINING' })

          if (env.data?.loss != null) {
            const current = readTraining()
            const hist = [...current.lossHistory, {
              step: (current.lossHistory[current.lossHistory.length - 1]?.step ?? 0) + 1,
              loss: env.data.loss,
            }]
            writeTraining({ loss: env.data.loss, lossHistory: hist.length > 200 ? hist.slice(-200) : hist })
          }
          if (env.data?.eval_loss != null) {
            const current = readTraining()
            const hist = [...current.lossHistory, {
              step: env.data?.step ?? current.lossHistory.length,
              loss: env.data.eval_loss,
              isEval: true,
            }]
            writeTraining({ lossHistory: hist.length > 200 ? hist.slice(-200) : hist })
          }
          if (env.data?.progress != null) writeTraining({ progress: env.data.progress })
          if (env.data?.global_step != null) writeTraining({ globalStep: env.data.global_step })
          if (env.data?.total_steps != null) writeTraining({ totalSteps: env.data.total_steps })
          if (env.data?.eta_s != null) writeTraining({ eta: env.data.eta_s })
          if (env.data?.steps_per_sec != null) writeTraining({ stepsPerSec: env.data.steps_per_sec })
          if (env.data?.elapsed_s != null) writeTraining({ elapsedSeconds: env.data.elapsed_s })
          if (env.meta?.epoch != null) writeTraining({ epoch: env.meta.epoch })
          if (env.meta?.total_epochs != null) writeTraining({ totalEpochs: env.meta.total_epochs })
          if (env.message) writeTraining({ message: env.message })
          if (env.data?.eval_report) writeTraining({ evalResult: env.data.eval_report })

          if (env.status === 'complete') {
            closeStream()
            writeTraining({ phase: 'complete', checkpoint: env.data?.checkpoint ?? null, finalLoss: env.data?.final_loss ?? null })
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
    }).catch(() => addToast('Failed to start training', 'error'))
  }, [closeStream])

  useEffect(() => () => closeStream(), [closeStream])

  return { startSSETraining, closeStream }
}
