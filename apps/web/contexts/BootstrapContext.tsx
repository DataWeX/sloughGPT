'use client'

// ═══════════════════════════════════════════════════════════════════════════
// BootstrapContext — one GET /bootstrap per app load, root-mounted
// ═══════════════════════════════════════════════════════════════════════════
//
// The provider replaces the landing journey's ~24-call boot fan-out: one
// fetch serves the landing page, ModelContext, CommandPalette, and the
// consciousness quick actions from typed slices.
//
// While the server is still booting (startup_progress.phase !== 'ready')
// it re-fetches the whole payload every 3s — one loop replaces the old
// startup-progress poll AND picks up members as startup phases register
// them (late members simply transition null → data in a later payload).
// Once the server reports 'ready' the loop stops for the life of the
// process; live model state keeps flowing over the existing SSE stream.
//
// The startup_progress slice is also fed into the zustand modelReadiness
// store so StatusBar/TrainingStatus keep their existing consumers.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { bootstrapController, type BootstrapPayload } from '@/lib/bootstrap-controller'
import { setModelReadiness } from '@/lib/store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('bootstrap')

/** Poll cadence while the server reports a non-ready startup phase. */
const NOT_READY_POLL_MS = 3000

export interface BootstrapContextValue {
  /** The payload, or null until the first response lands (or if every
   *  attempt failed — consumers render skeletons / degraded states). */
  data: BootstrapPayload | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

const noopRefresh = async () => {}
const FALLBACK: BootstrapContextValue = {
  data: null,
  loading: true,
  error: null,
  refresh: noopRefresh,
}

const BootstrapContext = createContext<BootstrapContextValue | null>(null)

export function BootstrapProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<BootstrapPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const aliveRef = useRef(true)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const load = useCallback(async () => {
    try {
      const payload = await bootstrapController.get()
      if (!aliveRef.current) return
      setData(payload)
      setError(null)

      const sp = payload.members.startup_progress
      if (sp) {
        setModelReadiness({
          ready: sp.phase === 'ready',
          phase: sp.phase,
          step: sp.step,
          total: sp.total,
          message: sp.message,
        })
        // Startup phases are process-lifetime and end at 'ready' — keep
        // polling the whole payload until then (late members appear in
        // later payloads), stop for good once it reports ready.
        if (sp.phase !== 'ready') {
          if (timerRef.current) clearTimeout(timerRef.current)
          timerRef.current = setTimeout(() => {
            timerRef.current = null
            void load()
          }, NOT_READY_POLL_MS)
        }
      }
    } catch (e) {
      if (!aliveRef.current) return
      const message = e instanceof Error ? e.message : String(e)
      _log.warning('bootstrap fetch failed', { exception: message })
      // No retry loop against a dead server: the SSE health stream (with
      // its own fallback poll) owns offline detection and recovery UX.
      setError(message)
    }
  }, [])

  useEffect(() => {
    aliveRef.current = true
    void load()
    return () => {
      aliveRef.current = false
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }
  }, [load])

  const refresh = useCallback(async () => {
    await load()
  }, [load])

  const value = useMemo<BootstrapContextValue>(
    () => ({ data, loading: data === null, error, refresh }),
    [data, error, refresh],
  )

  return <BootstrapContext.Provider value={value}>{children}</BootstrapContext.Provider>
}

/**
 * Read the boot payload. Tolerates a missing provider (tests and any
 * surface mounting a subtree standalone): members are nullable by design,
 * so consumers degrade to their skeleton state rather than crashing.
 */
export function useBootstrap(): BootstrapContextValue {
  return useContext(BootstrapContext) ?? FALLBACK
}
