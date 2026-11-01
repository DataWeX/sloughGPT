'use client'

// Stateful core of the VM console page, extracted from page.tsx by the
// page-decomposition refactor: all state, chatDB hydration/persistence, the
// training-job poller, and the run/launch handlers live here — the page and
// its components stay presentational.

import { useState, useCallback, useRef, useEffect } from 'react'
import { vmController, type VMRunResult, type VMTrainingJob } from '@/lib/vm-controller'
import { datasetController } from '@/lib/dataset-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import { chatDB } from '@/lib/db'
import {
  DEFAULT_MAX_STEPS,
  DEFAULT_PROGRAMS,
  DEFAULT_TRAIN_CONFIG,
  buildTrainSource,
  clampSteps,
  clampTrainConfig,
  type TrainConfig,
} from './vm-data'

async function loadRole(): Promise<string> {
  try {
    const entry = await chatDB.getKV<string>('vm-role')
    const saved = entry ?? 'user'
    if (saved === 'admin' || saved === 'kernel' || saved === 'user') return saved
  } catch {
    /* storage unavailable — default to user */
  }
  return 'user'
}

async function loadMaxSteps(): Promise<number> {
  try {
    const entry = await chatDB.getKV<number>('vm-max-steps')
    const saved = Number(entry ?? NaN)
    if (Number.isFinite(saved)) return clampSteps(saved)
  } catch {
    /* storage unavailable — default steps */
  }
  return DEFAULT_MAX_STEPS
}

async function loadTrainConfig(): Promise<TrainConfig> {
  try {
    const entry = await chatDB.getKV<TrainConfig>('vm-train-config')
    if (entry) return clampTrainConfig(entry)
  } catch {
    /* storage unavailable — default config */
  }
  return DEFAULT_TRAIN_CONFIG
}

export function useVmPage() {
  const [vmMode, setVmMode] = useState<'assembly' | 'browser'>('assembly')
  const [source, setSourceRaw] = useState(DEFAULT_PROGRAMS.hello)
  const [result, setResult] = useState<VMRunResult | null>(null)
  const [running, setRunning] = useState(false)
  const [maxSteps, setMaxStepsRaw] = useState(DEFAULT_MAX_STEPS)
  const [debug, setDebug] = useState(false)
  const [keyboardInput, setKeyboardInput] = useState('')
  const [role, setRoleRaw] = useState('user')
  const [hydrated, setHydrated] = useState(false)
  const [showRef, setShowRef] = useState(false)
  const [copied, setCopied] = useState(false)
  const [trainingJob, setTrainingJob] = useState<VMTrainingJob | null>(null)
  const [trainConfig, setTrainConfigRaw] = useState<TrainConfig>(DEFAULT_TRAIN_CONFIG)
  const [datasetNames, setDatasetNames] = useState<string[]>([])
  const [customDataset, setCustomDataset] = useState(false)
  const [launchedJob, setLaunchedJob] = useState<number | null>(null)

  // loadState() hydrates from chatDB asynchronously. Edits made before it
  // completes must not be clobbered by it: the user-facing setters below mark
  // the field as touched, and loadState skips touched fields.
  const sourceTouched = useRef(false)
  const stepsTouched = useRef(false)
  const roleTouched = useRef(false)
  const trainTouched = useRef(false)
  const setSource = useCallback<React.Dispatch<React.SetStateAction<string>>>((v) => {
    sourceTouched.current = true
    setSourceRaw(v)
  }, [])
  const setMaxSteps = useCallback<React.Dispatch<React.SetStateAction<number>>>((v) => {
    stepsTouched.current = true
    setMaxStepsRaw(v)
  }, [])
  const setRole = useCallback<React.Dispatch<React.SetStateAction<string>>>((v) => {
    roleTouched.current = true
    setRoleRaw(v)
  }, [])
  const setTrainConfig = useCallback<React.Dispatch<React.SetStateAction<TrainConfig>>>((v) => {
    trainTouched.current = true
    setTrainConfigRaw(v)
  }, [])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const trainingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
      if (trainingTimerRef.current) clearInterval(trainingTimerRef.current)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    datasetController
      .list()
      .then((list) => {
        if (cancelled) return
        setDatasetNames(list.map((d) => d.name).filter(Boolean))
      })
      .catch(() => {
        // fall back to the free-text dataset input
      })
    return () => {
      cancelled = true
    }
  }, [])

  const trainingJobId = result?.training_job_id ?? null
  const permissionDenied = !!(
    result && result.registers.some((r) => r.name === 'EAX' && r.hex.toLowerCase() === '0xfffffffe')
  )

  const pollTrainingJob = useCallback(async (): Promise<VMTrainingJob | null> => {
    if (trainingJobId == null) return null
    try {
      const job = await vmController.trainingJob(trainingJobId)
      setTrainingJob(job)
      return job
    } catch (err: unknown) {
      setTrainingJob({
        job_id: trainingJobId,
        api_job_id: '',
        status: 'error',
        progress: 0,
        error: extractErrorMessage(err),
      })
      return null
    }
  }, [trainingJobId])

  const handleStopTraining = useCallback(async () => {
    if (trainingJobId == null) return
    try {
      await vmController.stopTrainingJob(trainingJobId)
    } catch (err: unknown) {
      setTrainingJob({
        job_id: trainingJobId,
        api_job_id: '',
        status: 'error',
        progress: 0,
        error: extractErrorMessage(err),
      })
      return
    }
    await pollTrainingJob()
  }, [trainingJobId, pollTrainingJob])

  useEffect(() => {
    if (trainingJobId == null) {
      setTrainingJob(null)
      return
    }
    let disposed = false
    const terminal = new Set([
      'completed',
      'failed',
      'cancelled',
      'not_found',
      'interrupted',
      'error',
    ])
    const poll = async () => {
      const job = await pollTrainingJob()
      if (disposed || !job) return
      if (terminal.has(job.status)) {
        if (trainingTimerRef.current) clearInterval(trainingTimerRef.current)
        trainingTimerRef.current = null
      }
    }
    poll()
    trainingTimerRef.current = setInterval(poll, 3000)
    return () => {
      disposed = true
      if (trainingTimerRef.current) clearInterval(trainingTimerRef.current)
      trainingTimerRef.current = null
    }
  }, [trainingJobId, pollTrainingJob])

  // The three GET-backed fields, loaded from their own controllers. Guarded by
  // the *Touched refs, so a refresh can never overwrite a field the user has
  // edited — untouched fields simply pick up the fresh server value. Shared by
  // the mount effect and the 'r' shortcut: refresh re-runs these GETs instead
  // of reloading the document, which used to destroy the training timer and
  // every form field. Training timers, *Touched refs and form state are never
  // touched here.
  const loadVmData = useCallback(async () => {
    const [role, maxSteps, trainConfigData] = await Promise.all([
      loadRole(),
      loadMaxSteps(),
      loadTrainConfig(),
    ])
    if (!roleTouched.current) setRoleRaw(role)
    if (!stepsTouched.current) setMaxStepsRaw(maxSteps)
    if (!trainTouched.current) setTrainConfigRaw(trainConfigData)
    setHydrated(true)
  }, [])

  // Load from URL hash (#code=base64) or persisted state after hydration.
  // Mount-only: the hash is a one-shot deep-link, not something to re-read on
  // every refresh.
  useEffect(() => {
    let sourceLoaded = false
    try {
      const hash = window.location.hash
      if (hash.startsWith('#code=')) {
        const decoded = atob(hash.slice(6))
        setSourceRaw(decoded)
        sourceLoaded = true
      }
    } catch {
      /* malformed hash — fall through to chatDB */
    }

    const loadSource = async () => {
      if (!sourceLoaded && !sourceTouched.current) {
        const saved = await chatDB.getKV<string>('vm-source')
        if (saved) setSourceRaw(saved)
      }
    }

    // A rejected KV read must not leave the page stuck on the loading
    // skeleton (hydrated gates both rendering and the save effects).
    Promise.allSettled([loadVmData(), loadSource()]).then((results) => {
      if (results.some((r) => r.status === 'rejected')) setHydrated(true)
    })
  }, [loadVmData])

  // 'r' re-runs this page's own idempotent GET loads — never the document
  // (which destroyed the training timer and every form field).
  useRefreshShortcut(() => {
    loadVmData().catch(() => setHydrated(true))
  })

  // Save source to chatDB on change.
  // Touched fields save immediately even before hydration (the hydration load
  // skips touched fields, so writing early is safe); untouched fields wait so
  // a default value can never overwrite the stored one on mount.
  useEffect(() => {
    if (!hydrated && !sourceTouched.current) return
    chatDB.setKV('vm-source', source).catch(() => {})
  }, [source, hydrated])

  // Save role and steps to chatDB on change
  useEffect(() => {
    if (!hydrated && !roleTouched.current) return
    chatDB.setKV('vm-role', role).catch(() => {})
  }, [role, hydrated])

  useEffect(() => {
    if (!hydrated && !stepsTouched.current) return
    chatDB.setKV('vm-max-steps', maxSteps).catch(() => {})
  }, [maxSteps, hydrated])

  // Save the training launch config to chatDB on change
  useEffect(() => {
    if (!hydrated && !trainTouched.current) return
    chatDB.setKV('vm-train-config', clampTrainConfig(trainConfig)).catch(() => {})
  }, [trainConfig, hydrated])

  const handleRun = useCallback(
    async (step?: boolean, srcOverride?: string): Promise<VMRunResult | null> => {
      setRunning(true)
      setResult(null)
      const src = srcOverride ?? source
      try {
        const res = await vmController.run(src, {
          maxSteps: step ? 1 : clampSteps(maxSteps),
          role,
          debug,
          keyboardInput: keyboardInput || undefined,
        })
        setResult(res)
        return res
      } catch (err: unknown) {
        setResult({
          success: false,
          exit_code: -1,
          steps_executed: 0,
          elapsed_ms: 0,
          output: '',
          registers: [],
          eip: 0,
          eip_hex: '0x0',
          status: 'error',
          error: extractErrorMessage(err),
        })
        return null
      } finally {
        setRunning(false)
      }
    },
    [source, maxSteps, role, debug],
  )

  const handleLaunchTraining = useCallback(async () => {
    const src = buildTrainSource(clampTrainConfig(trainConfig))
    setSource(src)
    const res = await handleRun(false, src)
    const eax = res?.registers.find((r) => r.name === 'EAX')?.value
    setLaunchedJob(eax != null && eax >= 1 ? eax : null)
  }, [trainConfig, handleRun, setSource])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        handleRun()
      }
    },
    [handleRun],
  )

  return {
    vmMode,
    setVmMode,
    source,
    setSource,
    result,
    setResult,
    running,
    maxSteps,
    setMaxSteps,
    debug,
    setDebug,
    keyboardInput,
    setKeyboardInput,
    role,
    setRole,
    showRef,
    setShowRef,
    copied,
    setCopied,
    copiedTimerRef,
    textareaRef,
    trainingJob,
    trainConfig,
    setTrainConfig,
    datasetNames,
    customDataset,
    setCustomDataset,
    launchedJob,
    setLaunchedJob,
    permissionDenied,
    handleRun,
    handleStopTraining,
    handleLaunchTraining,
    handleKeyDown,
  }
}

export type VmPageState = ReturnType<typeof useVmPage>
