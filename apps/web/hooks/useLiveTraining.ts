'use client'

import { useEffect, useRef } from 'react'
import { createSSEStream, type SSEEnvelope } from '@/lib/sse-client'
import { readTraining, writeTraining, type TrainingShellState } from '@/lib/app-shell'
import { trainingJobsController, type TrainingJob } from '@/lib/controllers'

const MAX_LOSS_HISTORY = 200
const FALLBACK_POLL_MS = 5000

// ── Batched writeTraining — coalesces rapid progress events ─────────
// Training progress events arrive every 1-5 steps (~50-200ms apart).
// Without batching, each event triggers a Zustand write → React re-render
// of every subscriber (useTrainingSession, TrainingPipeline, LossChart).
// This queue collects patches and flushes at 60fps via rAF.

let _pendingPatch: Partial<TrainingShellState> | null = null
let _rafId: number | null = null
let _flushScheduled = false

function scheduleFlush() {
  if (_flushScheduled) return
  _flushScheduled = true
  _rafId = requestAnimationFrame(flushPatch)
}

function flushPatch() {
  _flushScheduled = false
  _rafId = null
  if (_pendingPatch) {
    const patch = _pendingPatch
    _pendingPatch = null
    writeTraining(patch)
  }
}

function batchWriteTraining(patch: Partial<TrainingShellState>) {
  // Terminal events (complete/error) flush immediately — no batching
  if (patch.phase === 'complete' || patch.phase === 'error') {
    if (_rafId != null) cancelAnimationFrame(_rafId)
    _flushScheduled = false
    _rafId = null
    _pendingPatch = null
    writeTraining(patch)
    return
  }

  // Merge into pending patch — latest value wins for each key
  if (_pendingPatch) {
    Object.assign(_pendingPatch, patch)
  } else {
    _pendingPatch = { ...patch }
  }
  scheduleFlush()
}

// ── Singleton SSE connection ────────────────────────────────────────

let _stream: ReturnType<typeof createSSEStream> | null = null
let _fallbackTimer: ReturnType<typeof setInterval> | null = null
let _receivedInit = false
let _stopped = true
let _refCount = 0
let _pollingJobId: string | null = null

function stopFallback() {
  if (_fallbackTimer) {
    clearInterval(_fallbackTimer)
    _fallbackTimer = null
  }
  _pollingJobId = null
}

function applyJobToShell(job: TrainingJob) {
  const current = readTraining()
  if (job.status === 'running') {
    const patch: Partial<TrainingShellState> = {
      progress: job.progress ?? current.progress,
      loss: job.loss ?? job.train_loss ?? current.loss,
      epoch: job.current_epoch ?? current.epoch,
      totalEpochs: job.epochs ?? current.totalEpochs,
      globalStep: job.global_step ?? current.globalStep,
      totalSteps: job.total_steps ?? current.totalSteps,
      stepsPerSec: job.steps_per_sec ?? current.stepsPerSec,
      eta: job.eta_s ?? current.eta,
      elapsedSeconds: job.elapsed_s ?? current.elapsedSeconds,
      avgQuality: job.avg_quality ?? current.avgQuality,
    }
    if (job.loss != null || job.train_loss != null) {
      const loss = (job.loss ?? job.train_loss) as number
      const step = job.global_step ?? current.globalStep
      const hist = [...current.lossHistory, { step: step || current.lossHistory.length, loss }]
      patch.lossHistory = hist.length > MAX_LOSS_HISTORY ? hist.slice(-MAX_LOSS_HISTORY) : hist
    }
    batchWriteTraining(patch)
  } else if (job.status === 'completed') {
    batchWriteTraining({
      phase: 'complete', progress: 100,
      checkpoint: job.checkpoint ?? null,
      finalLoss: job.loss ?? job.train_loss ?? null,
      modelPath: job.output_dir ?? null,
      avgQuality: (job.metrics as Record<string, unknown> | undefined)?.avg_quality as number ?? null,
    })
  } else if (job.status === 'failed') {
    batchWriteTraining({ phase: 'error', error: job.error || 'Could not training' })
  }
}

function startFallbackPoll(jobId: string) {
  stopFallback()
  _pollingJobId = jobId
  _fallbackTimer = setInterval(async () => {
    try {
      const job = await trainingJobsController.get(jobId)
      if (!job) { stopFallback(); return }
      applyJobToShell(job)
      if (job.status !== 'running' && job.status !== 'queued') {
        stopFallback()
      }
    } catch {
      // Transient error — keep polling
    }
  }, FALLBACK_POLL_MS)
}

function onEvent(envelope: SSEEnvelope) {
  if (envelope.stream !== 'training') return
  _receivedInit = true
  stopFallback()

  const event = envelope.phase?.toLowerCase()
  const data = envelope.data as Record<string, unknown>
  const jobId = (data.job_id as string) || null

  if (event === 'init') {
    const jobs = data.jobs as Record<string, TrainingJob> | undefined
    if (jobs) {
      const activeJobId = Object.keys(jobs)[0]
      const activeJob = Object.values(jobs)[0] as TrainingJob | undefined
      if (activeJob && activeJob.status === 'running') {
        batchWriteTraining({ jobId: activeJobId, phase: 'TRAINING' })
        applyJobToShell(activeJob)
      }
    }
    return
  }

  if (event === 'progress' && jobId) {
    const current = readTraining()
    const d = data as Record<string, number | string | null | undefined>
    const patch: Partial<TrainingShellState> = { jobId }
    if (d.progress != null) patch.progress = Number(d.progress)
    if (d.epoch != null) patch.epoch = Number(d.epoch)
    if (d.global_step != null) patch.globalStep = Number(d.global_step)
    if (d.total_steps != null) patch.totalSteps = Number(d.total_steps)
    if (d.train_loss != null) {
      const loss = Number(d.train_loss)
      patch.loss = loss
      const step = Number(d.global_step) || current.globalStep
      const hist = [...current.lossHistory, { step: step || current.lossHistory.length, loss }]
      patch.lossHistory = hist.length > MAX_LOSS_HISTORY ? hist.slice(-MAX_LOSS_HISTORY) : hist
    }
    if (d.eval_loss != null) patch.loss = Number(d.eval_loss)
    if (d.eta_s != null) patch.eta = Number(d.eta_s)
    if (d.elapsed_s != null) patch.elapsedSeconds = Number(d.elapsed_s)
    if (Object.keys(patch).length > 1) batchWriteTraining(patch)
    return
  }

  if (event === 'started' && jobId) {
    batchWriteTraining({ jobId, phase: 'TRAINING', progress: 0 })
    return
  }

  if (event === 'completed' && jobId) {
    const current = readTraining()
    batchWriteTraining({
      phase: 'complete', progress: 100,
      checkpoint: (data.checkpoint as string) ?? current.checkpoint,
      finalLoss: data.loss != null ? Number(data.loss) : current.finalLoss,
      modelPath: (data.model_path as string) ?? current.modelPath,
    })
    return
  }

  if (event === 'failed' && jobId) {
    batchWriteTraining({ phase: 'error', error: (data.error as string) || 'Could not training' })
    return
  }
}

function onClose() {
  if (!_stopped) {
    const current = readTraining()
    if (current.jobId && current.phase === 'TRAINING') {
      startFallbackPoll(current.jobId)
    }
  }
}

function onError() {
  if (!_stopped) {
    const current = readTraining()
    if (current.jobId && current.phase === 'TRAINING') {
      startFallbackPoll(current.jobId)
    }
  }
}

/**
 * Initialize the singleton training SSE stream.
 * Safe to call multiple times — idempotent.
 */
export function initTrainingStream(): () => void {
  _refCount++
  if (_stream) return () => decrementRef()

  _stopped = false
  _receivedInit = false

  _stream = createSSEStream({
    url: '/training/stream',
    onEvent,
    onOpen: () => {},
    onClose,
    onError,
    reconnect: true,
    maxReconnects: Infinity,
    baseReconnectMs: 3000,
    maxReconnectMs: 15_000,
  })
  _stream.start()

  const graceTimer = setTimeout(() => {
    if (!_receivedInit && !_stopped) {
      const current = readTraining()
      if (current.jobId && current.phase === 'TRAINING') {
        startFallbackPoll(current.jobId)
      }
    }
  }, 5000)

  return () => {
    clearTimeout(graceTimer)
    decrementRef()
  }
}

function decrementRef() {
  _refCount--
  if (_refCount <= 0) {
    _refCount = 0
    _stopped = true
    _stream?.stop()
    _stream = null
    stopFallback()
    // Flush any pending batch on cleanup
    if (_pendingPatch) {
      const patch = _pendingPatch
      _pendingPatch = null
      writeTraining(patch)
    }
    if (_rafId != null) cancelAnimationFrame(_rafId)
    _flushScheduled = false
  }
}

// ── Hook ────────────────────────────────────────────────────────────

/**
 * SSE-backed training hook — replaces HTTP polling with real-time push.
 *
 * Uses a singleton SSE connection to `/training/stream`. Multiple
 * components can use this hook simultaneously.
 *
 * Progress events are batched via requestAnimationFrame to avoid
 * triggering a React re-render on every training step.
 *
 * Usage:
 *   const { isTraining, connected } = useLiveTraining()
 */
export function useLiveTraining() {
  const cleanupRef = useRef<(() => void) | null>(null)
  useEffect(() => {
    cleanupRef.current = initTrainingStream()
    return () => { cleanupRef.current?.(); cleanupRef.current = null }
  }, [])

  return {
    /** Whether training is currently active */
    isTraining: readTraining().phase === 'TRAINING',
    /** SSE connection status */
    connected: _stream?.connected ?? false,
  }
}
