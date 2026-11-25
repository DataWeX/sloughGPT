/**
 * Pure run-state reducer for the factory-floor view.
 *
 * Contract source: docs/routers.md → "Run-stream event vocabulary", pinned by
 * TestExecuteStreamVocabulary / TestOrchestrateVocabulary (apps/api) and
 * asserted here (vitest). One reducer, two inputs — orchestrate SSE frames and
 * GET /agents/runs records — so live stations and polled history can never
 * disagree (the two-writer defect the ux/agents journey bans).
 *
 * No I/O, no store, no React: `runReducer` is a fold over frames,
 * `hydrateRun` folds a persisted run record into the same shape.
 */
import type { OrchestrateTask } from '@/lib/agents-controller'

export type RunPhase =
  | 'IDLE'
  | 'START'
  | 'TOOL_CALL'
  | 'TOOL_RESULT'
  | 'FINAL'
  | 'COMPLETE'
  | 'ERROR'
  | 'PLAN'
  | 'EXECUTE'
  | 'COMPOSE'

/** Phases where the run is actively streaming — hydration must not clobber them. */
const LIVE_PHASES: ReadonlySet<string> = new Set([
  'START',
  'TOOL_CALL',
  'TOOL_RESULT',
  'FINAL',
  'PLAN',
  'EXECUTE',
  'COMPOSE',
])

export type StationStatus = 'queued' | 'live' | 'done' | 'failed'

export interface StationState {
  agent: string
  description: string
  status: StationStatus
}

/** The minimal SSE frame shape the reducer consumes (stream/phase/status/data). */
export interface RunFrame {
  stream?: string
  phase?: string
  status?: string
  data?: Record<string, unknown>
  error?: string
}

export interface RunState {
  /** Last phase seen ('' before the first frame). */
  phase: string
  /** EXECUTE progress: current level / total levels (0/0 until announced). */
  level: number
  totalLevels: number
  /**
   * Task status in SERVER vocabulary (pending | live | completed | failed) —
   * what GET /agents/runs records and the orchestration card render.
   * `stations` below is the floor's own vocabulary; the reducer writes both.
   */
  taskStatuses: Record<string, string>
  /** Floor stations keyed by task id (queued | live | done | failed). */
  stations: Record<string, StationState>
  /** Task list from PLAN success / run record (labels, order, depends_on). */
  tasks: OrchestrateTask[]
  /** Composed final answer (COMPOSE payload). */
  response: string | null
  /** Sticky failure message. */
  error: string | null
  /** Terminal flag: COMPLETE or ERROR seen (or run record terminal). */
  done: boolean
}

export const initialRunState: RunState = {
  phase: '',
  level: 0,
  totalLevels: 0,
  taskStatuses: {},
  stations: {},
  tasks: [],
  response: null,
  error: null,
  done: false,
}

/** True when the state is mid-stream (hydration must not overwrite it). */
export function isLive(state: RunState): boolean {
  return LIVE_PHASES.has(state.phase)
}

function stationFromTask(task: {
  id?: unknown
  agent?: unknown
  description?: unknown
  status?: unknown
}): [string, StationState] | null {
  const id = typeof task.id === 'string' ? task.id : null
  if (!id) return null
  return [
    id,
    {
      agent: typeof task.agent === 'string' ? task.agent : '',
      description: typeof task.description === 'string' ? task.description : '',
      status: toStationStatus(typeof task.status === 'string' ? task.status : 'pending'),
    },
  ]
}

/** Server task status → station status. Unknown values stay queued (safe). */
function toStationStatus(status: string): StationStatus {
  if (status === 'completed' || status === 'success') return 'done'
  if (status === 'failed' || status === 'error') return 'failed'
  if (status === 'running' || status === 'live' || status === 'in_progress') return 'live'
  return 'queued'
}

/**
 * Fold one SSE frame into the state.
 *
 * Unknown phases pass through (the floor ignores what it does not know);
 * terminal states are sticky until the next PLAN; task-level transitions
 * update both `stations` and the legacy `taskStatuses` mirror.
 */
export function runReducer(state: RunState, frame: RunFrame): RunState {
  const phase = frame.phase ?? ''
  const status = frame.status ?? ''
  const data = frame.data ?? {}

  // A new orchestration un-sticks the previous terminal state.
  const fresh: RunState =
    phase === 'PLAN' && status === 'working' ? { ...initialRunState } : state

  // Sticky terminal: ignore frames after COMPLETE/ERROR (except a fresh PLAN).
  if (fresh.done && !(phase === 'PLAN' && status === 'working')) return fresh

  const next: RunState = {
    ...fresh,
    phase: phase || fresh.phase,
    taskStatuses: { ...fresh.taskStatuses },
    stations: { ...fresh.stations },
    tasks: fresh.tasks,
  }

  switch (phase) {
    case 'PLAN': {
      if (status === 'success') {
        const tasks = Array.isArray(data.tasks) ? data.tasks : []
        const parsed: OrchestrateTask[] = []
        for (const t of tasks) {
          const entry = stationFromTask(t as { id?: unknown })
          if (entry) {
            next.stations[entry[0]] = entry[1]
            const raw = t as { status?: unknown }
            next.taskStatuses[entry[0]] =
              typeof raw.status === 'string' && raw.status ? raw.status : 'pending'
            parsed.push(t as OrchestrateTask)
          }
        }
        next.tasks = parsed
      } else if (status === 'error') {
        next.done = true
        next.error = frame.error ?? (typeof data.error === 'string' ? data.error : 'Could not plan this goal')
      }
      break
    }
    case 'EXECUTE': {
      if (typeof data.levels === 'number') next.totalLevels = data.levels
      if (typeof data.level === 'number') next.level = data.level
      const taskId = typeof data.task_id === 'string' ? data.task_id : null
      if (taskId) {
        const prev = next.stations[taskId]
        const station: StationState = {
          agent: typeof data.agent === 'string' ? data.agent : prev?.agent ?? '',
          description: typeof data.description === 'string' ? data.description : prev?.description ?? '',
          status: status === 'error' ? 'failed' : status === 'success' ? 'done' : 'live',
        }
        next.stations[taskId] = station
        next.taskStatuses[taskId] =
          status === 'error' ? 'failed' : status === 'success' ? 'completed' : 'live'
      } else if (status === 'error' && typeof data.error === 'string') {
        next.done = true
        next.error = data.error
      }
      break
    }
    case 'COMPOSE': {
      if (status === 'success' && typeof data.response === 'string') next.response = data.response
      break
    }
    case 'COMPLETE': {
      next.done = true
      break
    }
    case 'ERROR': {
      next.done = true
      next.error = frame.error ?? (typeof data.error === 'string' ? data.error : 'Run failed')
      break
    }
    default:
      // Unknown phase: pass through as-is (contract: ignore, do not crash).
      break
  }

  return next
}

/** Fold a batch of frames (replaying a captured stream). */
export function applyFrames(state: RunState, frames: RunFrame[]): RunState {
  return frames.reduce(runReducer, state)
}

/**
 * Fold a persisted run record (GET /agents/runs) into the same state shape.
 *
 * Guard: while a stream is live, hydration never overwrites live stations nor
 * flips the run to terminal — a poll must not clobber in-flight truth (the
 * two-writer defect). Everything else (completed/failed records, queued
 * stations) hydrates normally so history looks right with no live stream.
 */
export interface RunRecord {
  status?: string
  response?: string | null
  error?: string | null
  tasks?: Array<{
    id?: string
    agent?: string
    description?: string
    status?: string
    result_preview?: string
    depends_on?: string[]
  }>
}

export function hydrateRun(state: RunState, run: RunRecord): RunState {
  const live = isLive(state)
  const stations: Record<string, StationState> = { ...state.stations }
  const taskStatuses: Record<string, string> = { ...state.taskStatuses }
  const tasks: OrchestrateTask[] = []

  for (const t of run.tasks ?? []) {
    const entry = stationFromTask(t)
    if (!entry) continue
    const [id, station] = entry
    tasks.push(t as OrchestrateTask)
    if (live && stations[id]?.status === 'live') continue // never clobber live truth
    stations[id] = station
    taskStatuses[id] = typeof t.status === 'string' && t.status ? t.status : 'pending'
  }

  const next: RunState = { ...state, stations, taskStatuses }

  if (live) return next // a stream owns phase/done/tasks until it ends
  next.tasks = tasks

  const status = run.status ?? ''
  if (status === 'completed') {
    next.done = true
    next.phase = 'COMPLETE'
    next.response = typeof run.response === 'string' ? run.response : next.response
    next.error = null
  } else if (status === 'failed') {
    next.done = true
    next.phase = 'ERROR'
    next.error = typeof run.error === 'string' ? run.error : 'Run failed'
  }
  return next
}
