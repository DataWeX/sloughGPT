/**
 * Factory-floor run-state reducer — red-first pin (ux/agents "Watch the Factory").
 *
 * The reducer is the whole point of the floor view: every frame from the
 * orchestrate stream and every record from GET /agents/runs must reduce to
 * one deterministic RunState, or the stations lie (the exact defect the
 * journey bans). Vocabulary source: docs/routers.md
 * "Run-stream event vocabulary".
 */
import { describe, expect, it } from 'vitest'
import {
  applyFrames,
  hydrateRun,
  initialRunState,
  runReducer,
} from './run-state'

const orch = (phase: string, status: string, data: Record<string, unknown> = {}, error?: string) => ({
  stream: 'agent-orchestrate',
  phase,
  status,
  data,
  ...(error ? { error } : {}),
})

describe('runReducer — orchestrate frames', () => {
  it('walks PLAN → EXECUTE → COMPOSE → COMPLETE', () => {
    let s = initialRunState
    s = runReducer(s, orch('PLAN', 'working'))
    expect(s.phase).toBe('PLAN')
    expect(s.done).toBe(false)

    s = runReducer(
      s,
      orch('PLAN', 'success', {
        tasks: [
          { id: '1', agent: 'researcher', description: 'research', status: 'pending' },
          { id: '2', agent: 'writer', description: 'write', status: 'pending' },
        ],
      }),
    )
    expect(s.stations['1']).toEqual({ agent: 'researcher', description: 'research', status: 'queued' })
    expect(s.stations['2'].status).toBe('queued')

    s = runReducer(s, orch('EXECUTE', 'working', { levels: 2, level: 1 }))
    expect(s.level).toBe(1)
    expect(s.totalLevels).toBe(2)

    s = runReducer(s, orch('EXECUTE', 'working', { task_id: '1', agent: 'researcher', description: 'research' }))
    expect(s.stations['1'].status).toBe('live')
    expect(s.taskStatuses['1']).toBe('live')

    s = runReducer(s, orch('EXECUTE', 'success', { task_id: '1', agent: 'researcher', description: 'research', result_preview: 'ok' }))
    expect(s.stations['1'].status).toBe('done')

    s = runReducer(s, orch('EXECUTE', 'error', { task_id: '2', agent: 'writer', description: 'write' }))
    expect(s.stations['2'].status).toBe('failed')

    s = runReducer(s, orch('COMPOSE', 'working'))
    expect(s.phase).toBe('COMPOSE')

    s = runReducer(s, orch('COMPLETE', 'complete'))
    expect(s.done).toBe(true)
    expect(s.response).toBeNull() // response arrives via its own data frame
  })

  it('captures COMPOSE payload as the response and errors as sticky failure', () => {
    let s = runReducer(initialRunState, orch('COMPOSE', 'working'))
    s = runReducer(s, orch('COMPOSE', 'success', { response: 'the answer' }))
    expect(s.response).toBe('the answer')

    s = runReducer(initialRunState, orch('ERROR', 'error', {}, 'boom'))
    expect(s.done).toBe(true)
    expect(s.phase).toBe('ERROR')
    expect(s.error).toBe('boom')

    // A later COMPLETE must not resurrect a failed run.
    const failed = runReducer(initialRunState, orch('ERROR', 'error', {}, 'boom'))
    expect(runReducer(failed, orch('COMPLETE', 'complete')).done).toBe(true)
    expect(runReducer(failed, orch('COMPLETE', 'complete')).phase).toBe('ERROR')
  })

  it('unknown phases pass through without corrupting stations', () => {
    const s = runReducer(initialRunState, { stream: 'agent', phase: 'WAT', status: 'working' })
    expect(s.phase).toBe('WAT')
    expect(s.stations).toEqual({})
    expect(s.done).toBe(false)
  })
})

describe('applyFrames — batch (hydration of a replayed stream)', () => {
  it('reduces a frame list to the same state as stepwise reduction', () => {
    const frames = [
      orch('PLAN', 'working'),
      orch('PLAN', 'success', { tasks: [{ id: '1', agent: 'a', description: 'd', status: 'pending' }] }),
      orch('EXECUTE', 'working', { task_id: '1', agent: 'a', description: 'd' }),
      orch('EXECUTE', 'success', { task_id: '1', agent: 'a', description: 'd', result_preview: 'x' }),
      orch('COMPOSE', 'working'),
      orch('COMPLETE', 'complete'),
    ]
    const batched = applyFrames(initialRunState, frames)
    const stepped = frames.reduce(runReducer, initialRunState)
    expect(batched).toEqual(stepped)
    expect(batched.done).toBe(true)
    expect(batched.stations['1'].status).toBe('done')
  })
})

describe('hydrateRun — GET /agents/runs → stations (no SSE available)', () => {
  it('maps a completed run to done stations', () => {
    const s = hydrateRun(initialRunState, {
      status: 'completed',
      response: 'final text',
      tasks: [
        { id: '1', agent: 'researcher', description: 'research', status: 'completed' },
        { id: '2', agent: 'writer', description: 'write', status: 'failed' },
      ],
    })
    expect(s.done).toBe(true)
    expect(s.phase).toBe('COMPLETE')
    expect(s.stations['1'].status).toBe('done')
    expect(s.stations['2'].status).toBe('failed')
    expect(s.response).toBe('final text')
  })

  it('maps a failed run to a sticky error with its message', () => {
    const s = hydrateRun(initialRunState, { status: 'failed', error: 'kaput', tasks: [] })
    expect(s.done).toBe(true)
    expect(s.phase).toBe('ERROR')
    expect(s.error).toBe('kaput')
  })

  it('leaves a running run live (not done) with queued/live stations', () => {
    const s = hydrateRun(initialRunState, {
      status: 'running',
      tasks: [
        { id: '1', agent: 'a', description: 'd', status: 'completed' },
        { id: '2', agent: 'b', description: 'e', status: 'running' },
        { id: '3', agent: 'c', description: 'f', status: 'pending' },
      ],
    })
    expect(s.done).toBe(false)
    expect(s.stations['1'].status).toBe('done')
    expect(s.stations['2'].status).toBe('live')
    expect(s.stations['3'].status).toBe('queued')
  })

  it('never lets hydration overwrite a live in-flight state with stale history', () => {
    // A poll that lands after a live frame must not clobber the live station.
    const live = runReducer(
      initialRunState,
      orch('EXECUTE', 'working', { task_id: '1', agent: 'a', description: 'd' }),
    )
    const hydrated = hydrateRun(live, { status: 'running', tasks: [{ id: '1', agent: 'a', description: 'd', status: 'pending' }] })
    // Hydration is only applied when there is no live phase — document that guard here:
    expect(hydrated.stations['1'].status).toBe('live')
  })
})
