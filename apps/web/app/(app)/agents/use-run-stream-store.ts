/**
 * Run-stream store — the single write authority for agent run state.
 *
 * Two writers used to exist (orchestrate SSE callbacks in hook-local state vs.
 * GET /agents/runs polling in `useAgentsWorkbench`), which is exactly the
 * ghost-activity defect banned by ux/agents "Watch the Factory". Now there is
 * one writer each way and one store:
 *
 *   SSE frames   → `consume`   (live truth, owns phase/done while streaming)
 *   GET /agents/runs → `hydrate` (history; a no-op for live stations — see
 *                      `hydrateRun`'s guard in run-state.ts)
 *
 * Reads: orchestration card + floor view (and anything else that wants run
 * truth). Pure reducer lives in ./run-state.ts (pinned by run-state.test.ts).
 */
import { create } from 'zustand'
import {
  applyFrames,
  hydrateRun,
  initialRunState,
  runReducer,
  type RunFrame,
  type RunRecord,
  type RunState,
} from './run-state'

export interface RunStreamState extends RunState {
  /** True while an SSE stream is driving the state (hydration is fenced out). */
  live: boolean
  /** Id of the run currently streaming / selected (null = none yet). */
  runId: string | null
  /** Frames seen this stream — cheap liveness signal for the floor header. */
  frameCount: number
}

export interface RunStreamActions {
  /** Reset and open a new stream (called before POST /agents/orchestrate). */
  beginStream: (runId?: string) => void
  /** Fold one SSE frame (the live writer). */
  consume: (frame: RunFrame) => void
  /** Fold a batch (replaying a captured stream — tests, reconnect replay). */
  consumeAll: (frames: RunFrame[]) => void
  /** Stream ended (COMPLETE/ERROR seen or connection closed). */
  endStream: () => void
  /** History writer: hydrate from GET /agents/runs (fenced while live). */
  hydrate: (run: RunRecord, runId?: string) => void
  /** Back to idle (leaving the page / switching selection). */
  reset: () => void
}

const initialState: RunStreamState = {
  ...initialRunState,
  live: false,
  runId: null,
  frameCount: 0,
}

export const useRunStreamStore = create<RunStreamState & RunStreamActions>()(
  (set) => ({
    ...initialState,

    beginStream: (runId) =>
      set({
        ...initialRunState,
        // Match the pre-store UX: the phase line reads "Planning…" from click.
        phase: 'PLAN',
        live: true,
        runId: runId ?? null,
        frameCount: 0,
      }),

    consume: (frame) =>
      set((s) => ({
        ...runReducer(s, frame),
        live: s.live,
        runId: s.runId,
        frameCount: s.frameCount + 1,
      })),

    consumeAll: (frames) =>
      set((s) => ({
        ...applyFrames(s, frames),
        live: s.live,
        runId: s.runId,
        frameCount: s.frameCount + frames.length,
      })),

    endStream: () => set({ live: false }),

    hydrate: (run, runId) =>
      set((s) => {
        // Switching runs: task ids ('1','2',…) are per-run, so a new run
        // starts from a clean slate instead of merging into the old stations.
        const base =
          runId && s.runId && runId !== s.runId
            ? { ...initialRunState, live: false, runId, frameCount: 0 }
            : s
        return { ...hydrateRun(base, run), runId: runId ?? s.runId }
      }),

    reset: () => set({ ...initialState }),
  }),
)

/**
 * Reconnect policy for POST /agents/orchestrate streams.
 *
 * Deliberately NOT the GET subscription default: replaying a POST re-fires the
 * orchestration (double cost, double side effects). So: three strikes, then
 * surface the failure — never silently re-run. Mirrors `useTrainingStream`'s
 * bounded-retry discipline, with the retry count reset on every frame (a
 * stream that made progress may retry once more; a dead one may not).
 */
export const MAX_STREAM_RETRIES = 3

/**
 * @param attempts retries already spent on this stream
 * @param frameCount frames the stream delivered before dropping
 * @returns whether another attempt is allowed
 *
 * Progress buys retries (up to 3): a stream that was streaming may resume.
 * No progress buys exactly one: a dead POST is never re-fired twice.
 */
export function shouldRetry(attempts: number, frameCount: number): boolean {
  if (attempts >= MAX_STREAM_RETRIES) return false
  return frameCount > 0 || attempts < 1
}
