import { describe, it, expect } from 'vitest'
import { BLOCK_NAMES, blocksFromDetailed, blocksFromSse, snapshotFromBlocks } from './health-blocks'
import type { LiveHealthSnapshot } from './health-blocks'
import type { DetailedHealth } from './system-controller'

// ---------------------------------------------------------------------------
// Fixtures — two hand-written payloads describing the SAME moment.
// Hand-written rather than derived from each other: a shared helper would
// transform both in the same wrong direction and the parity test would pass
// while the transports still disagreed in production.
// ---------------------------------------------------------------------------

const DIAGNOSES = [
  { check: 'errors', severity: 'ok', score: 100, message: '41 requests OK.' },
  { check: 'latency', severity: 'warning', score: 60, message: 'p95 above target.' },
]

const BANDWIDTH = {
  identity_bytes: 100_000,
  wire_bytes: 42_000,
  saved_bytes: 58_000,
  saved_pct: 58,
  compressed_responses: 12,
  identity_responses: 3,
  zstd_responses: 12,
  gzip_responses: 0,
}

const COLLECTIONS = {
  model_metrics: [
    { model: 'gpt2', count: 3, total_tokens: 1000, tokens_per_sec: 5, avg_tokens: 333 },
  ],
  model_events: [{ type: 'load', model: 'gpt2', detail: '', ts: 100 }],
  health_history: [{ score: 90, status: 'healthy', ts: 100 }],
  memory_history: [{ rss_mb: 300, virtual_mb: 400, system_percent: 55, ts: 100 }],
  rate_violations: [{ path: '/chat', count: 12, limit: 5, ts: 100 }],
  path_latencies: [{ path: '/inference/generate', avg_ms: 80.5, count: 5, p95_ms: 120 }],
  recent_errors: [
    { path: '/chat', method: 'POST', status: 500, message: 'boom', error_type: 'Err', ts: 100 },
  ],
}

const STARTUP = {
  stage: 'ready',
  stage_value: 2,
  elapsed_seconds: 5.5,
  model_progress: 1,
  model_progress_message: 'loading gpt2',
  errors: {},
  hooks: {
    load_model: {
      name: 'load_model',
      stage: 'critical',
      status: 'ok',
      duration_seconds: 1,
      error: null,
    },
  },
  stages: { critical: { hooks: ['load_model'], time: 1 } },
}

const TRAINING_POOL = { active_jobs: 1, max_workers: 4, total_tracked: 9 }

/** `/health/detailed` — nested, health_score carries summary + diagnoses. */
function makeDetailed(): DetailedHealth {
  return {
    status: 'healthy',
    uptime_seconds: 512.5,
    request_count: 42,
    error_count: 1,
    avg_latency_ms: 80.5,
    p95_latency_ms: 140,
    requests_per_minute: 4.5,
    ...COLLECTIONS,
    inference_count: 8,
    total_tokens: 12_000,
    tokens_per_sec: 12.5,
    avg_tokens_per_request: 160,
    health_score: {
      score: 90,
      status: 'healthy',
      summary: 'gpt2: p95 above target.',
      diagnoses: DIAGNOSES,
    },
    // HTTP's summary source: the human-readable status line, not the health
    // flow one-liner above. Deliberately different from SSE's — see the
    // KNOWN_TRANSPORT_DIVERGENCE note below.
    status_message: 'gpt2 loaded — ready',
    system: { cpu_percent: 45.2, memory_percent: 60.1, memory_available_mb: 8000 },
    model_loaded: true,
    model_loading: false,
    model_type: 'gpt2',
    device: 'cpu',
    soul: 'friendly',
    num_parameters: null, // phantom: get_detailed_health() never emits it
    inference: { is_inferencing: true, inference_count: 8 },
    quantization: null,
    training_pool: TRAINING_POOL,
    bandwidth: BANDWIDTH,
    startup_progress: STARTUP,
  } as unknown as DetailedHealth
}

/** `/health/stream` — flattened, health_score collapsed to an int. */
function makeSse(): Record<string, unknown> {
  return {
    status: 'healthy',
    uptime_seconds: 512.5,
    request_count: 42,
    error_count: 1,
    avg_latency_ms: 80.5,
    p95_latency_ms: 140,
    requests_per_minute: 4.5,
    ...COLLECTIONS,
    inference_count: 8,
    total_tokens: 12_000,
    tokens_per_sec: 12.5,
    avg_tokens_per_request: 160,
    health_score: 90,
    health_status: 'healthy',
    health_summary: 'gpt2: p95 above target.',
    diagnoses: DIAGNOSES,
    cpu_percent: 45.2,
    memory_percent: 60.1,
    model_loaded: true,
    model_loading: false,
    model_type: 'gpt2',
    device: 'cpu',
    soul: 'friendly',
    // Backend phantom: the SSE projection reads a TOP-LEVEL is_inferencing
    // that get_detailed_health() never produces, so SSE always sends false
    // even while the nested HTTP value is true. Preserved on purpose — see
    // the follow-up on kanban card 294818b7.
    is_inferencing: false,
    num_parameters: null, // same phantom, same follow-up
    quantization: null,
    training_pool: TRAINING_POOL,
    bandwidth: BANDWIDTH,
    startup_progress: STARTUP,
  }
}

/**
 * Fields the two transports are ALLOWED to disagree on. Each is a pinned
 * backend behaviour, not adapter drift — so anything not listed here failing
 * the parity test means a normalizer genuinely drifted.
 */
const KNOWN_TRANSPORT_DIVERGENCE: ReadonlyArray<keyof LiveHealthSnapshot> = [
  // HTTP reads status_message; SSE reads the health flow's one-liner.
  'health_summary',
  // Backend phantom: SSE always reports false, HTTP reports the real value.
  'is_inferencing',
]

function withoutKnownDivergence(snapshot: LiveHealthSnapshot): LiveHealthSnapshot {
  const clone = { ...snapshot }
  for (const key of KNOWN_TRANSPORT_DIVERGENCE) delete clone[key]
  return clone
}

// ---------------------------------------------------------------------------

describe('block coverage', () => {
  it('the HTTP adapter produces every declared block', () => {
    expect(Object.keys(blocksFromDetailed(makeDetailed())).sort()).toEqual([...BLOCK_NAMES].sort())
  })

  it('the SSE adapter produces every declared block', () => {
    expect(Object.keys(blocksFromSse(makeSse())).sort()).toEqual([...BLOCK_NAMES].sort())
  })

  it('still covers every block on an empty payload', () => {
    expect(Object.keys(blocksFromSse({})).sort()).toEqual([...BLOCK_NAMES].sort())
    expect(Object.keys(blocksFromDetailed({} as DetailedHealth)).sort()).toEqual(
      [...BLOCK_NAMES].sort(),
    )
  })
})

describe('transport parity', () => {
  it('HTTP and SSE flatten the same moment to the same snapshot', () => {
    const fromHttp = snapshotFromBlocks(blocksFromDetailed(makeDetailed()))
    const fromSse = snapshotFromBlocks(blocksFromSse(makeSse()))

    expect(withoutKnownDivergence(fromSse)).toEqual(withoutKnownDivergence(fromHttp))
  })

  it('agrees on every field that is not a pinned backend divergence', () => {
    const fromHttp = snapshotFromBlocks(blocksFromDetailed(makeDetailed()))
    const fromSse = snapshotFromBlocks(blocksFromSse(makeSse()))

    const disagreeing = Object.keys(fromHttp).filter(
      (key) =>
        !(KNOWN_TRANSPORT_DIVERGENCE as ReadonlyArray<string>).includes(key) &&
        JSON.stringify(fromHttp[key as keyof LiveHealthSnapshot]) !==
          JSON.stringify(fromSse[key as keyof LiveHealthSnapshot]),
    )
    expect(disagreeing).toEqual([])
  })

  it('pins exactly the known divergences — a new one fails this suite', () => {
    const fromHttp = snapshotFromBlocks(blocksFromDetailed(makeDetailed()))
    const fromSse = snapshotFromBlocks(blocksFromSse(makeSse()))

    const actuallyDiverging = (Object.keys(fromHttp) as Array<keyof LiveHealthSnapshot>).filter(
      (key) => JSON.stringify(fromHttp[key]) !== JSON.stringify(fromSse[key]),
    )
    expect(actuallyDiverging.sort()).toEqual([...KNOWN_TRANSPORT_DIVERGENCE].sort())
  })
})

describe('regression: fields each transport used to drop', () => {
  it('SSE now delivers gateway bandwidth to the card', () => {
    const snap = snapshotFromBlocks(blocksFromSse(makeSse()))
    expect(snap.bandwidth).toEqual(BANDWIDTH)
  })

  it('HTTP now delivers the health flow diagnoses', () => {
    const snap = snapshotFromBlocks(blocksFromDetailed(makeDetailed()))
    expect(snap.diagnoses).toEqual(DIAGNOSES)
    expect(snap.diagnoses).toHaveLength(2)
  })

  it('HTTP still defaults to no diagnoses when the score carries none', () => {
    const d = makeDetailed() as unknown as Record<string, unknown>
    ;(d.health_score as Record<string, unknown>) = { score: 90, status: 'healthy' }
    const snap = snapshotFromBlocks(blocksFromDetailed(d as unknown as DetailedHealth))
    expect(snap.diagnoses).toEqual([])
  })
})

describe('absent vs null vs zero', () => {
  it('bandwidth is absent when the payload never carried the key', () => {
    const d = makeDetailed() as unknown as Record<string, unknown>
    delete d.bandwidth
    const snap = snapshotFromBlocks(blocksFromDetailed(d as unknown as DetailedHealth))

    expect('bandwidth' in snap).toBe(false)
    expect(snap.bandwidth).toBeUndefined()
  })

  it('bandwidth is null when the gateway is present but unreachable', () => {
    const d = makeDetailed() as unknown as Record<string, unknown>
    d.bandwidth = null
    const snap = snapshotFromBlocks(blocksFromDetailed(d as unknown as DetailedHealth))

    expect('bandwidth' in snap).toBe(true)
    expect(snap.bandwidth).toBeNull()
  })

  it('bandwidth zero is a measurement, not missing data', () => {
    const sse = { ...makeSse(), bandwidth: { ...BANDWIDTH, identity_bytes: 0, saved_bytes: 0 } }
    const snap = snapshotFromBlocks(blocksFromSse(sse))

    expect(snap.bandwidth).not.toBeNull()
    expect(snap.bandwidth?.identity_bytes).toBe(0)
  })

  it('SSE treats an absent gateway key the same way as HTTP', () => {
    const sse = makeSse()
    delete sse.bandwidth
    const snap = snapshotFromBlocks(blocksFromSse(sse))

    expect('bandwidth' in snap).toBe(false)
  })

  it('zero counters stay zero instead of becoming null', () => {
    const snap = snapshotFromBlocks(
      blocksFromSse({ request_count: 0, error_count: 0, tokens_per_sec: 0 }),
    )
    expect(snap.request_count).toBe(0)
    expect(snap.error_count).toBe(0)
    expect(snap.tokens_per_sec).toBe(0)
  })

  it('unreadable gauges become null rather than a fake 0%', () => {
    const snap = snapshotFromBlocks(blocksFromSse({ cpu_percent: null, memory_percent: null }))
    expect(snap.cpu_percent).toBeNull()
    expect(snap.memory_percent).toBeNull()
  })
})

describe('per-transport reads that must not change shape', () => {
  it('HTTP reads is_inferencing from the nested inference object', () => {
    const snap = snapshotFromBlocks(blocksFromDetailed(makeDetailed()))
    expect(snap.is_inferencing).toBe(true)
  })

  it('SSE reads the flat is_inferencing key (preserved phantom)', () => {
    const nested = { inference: { is_inferencing: true } }
    const snap = snapshotFromBlocks(blocksFromSse(nested))
    expect(snap.is_inferencing).toBe(false)
  })

  it('HTTP reads cpu from system.*, SSE reads it flat', () => {
    expect(snapshotFromBlocks(blocksFromDetailed(makeDetailed())).cpu_percent).toBe(45.2)
    expect(snapshotFromBlocks(blocksFromSse({ cpu_percent: 45.2 })).cpu_percent).toBe(45.2)
    expect(
      snapshotFromBlocks(blocksFromSse({ system: { cpu_percent: 45.2 } })).cpu_percent,
    ).toBeNull()
  })

  it('HTTP summary comes from status_message', () => {
    const snap = snapshotFromBlocks(blocksFromDetailed(makeDetailed()))
    expect(snap.health_summary).toBe('gpt2 loaded — ready')
  })

  it('SSE summary comes from the health flow one-liner', () => {
    const snap = snapshotFromBlocks(blocksFromSse(makeSse()))
    expect(snap.health_summary).toBe('gpt2: p95 above target.')
  })
})

describe('startup overlay hardening', () => {
  it('HTTP resolves unknown stage to background once the model is loaded', () => {
    const snap = snapshotFromBlocks(blocksFromDetailed({ model_loaded: true } as DetailedHealth))
    expect(snap.startup_stage).toBe('background')
  })

  it('HTTP keeps unknown stage when no model is loaded', () => {
    const snap = snapshotFromBlocks(blocksFromDetailed({ model_loaded: false } as DetailedHealth))
    expect(snap.startup_stage).toBe('unknown')
  })

  it('SSE resolves unknown stage to background once the model is loaded', () => {
    const snap = snapshotFromBlocks(blocksFromSse({ model_loaded: true }))
    expect(snap.startup_stage).toBe('background')
  })

  it('prefers the nested startup_progress over flat startup_* keys', () => {
    const snap = snapshotFromBlocks(
      blocksFromSse({
        model_loaded: true,
        startup_stage: 'init',
        startup_progress: { ...STARTUP, stage: 'critical' },
      }),
    )
    expect(snap.startup_stage).toBe('critical')
    expect(snap.startup_stage_value).toBe(2)
    expect(snap.startup_hooks.load_model.status).toBe('ok')
  })

  it('falls back to flat startup keys when the object is missing', () => {
    const snap = snapshotFromBlocks(
      blocksFromSse({ model_loaded: true, startup_stage: 'ready', startup_elapsed: 7 }),
    )
    expect(snap.startup_stage).toBe('ready')
    expect(snap.startup_elapsed).toBe(7)
  })
})

describe('degraded payloads', () => {
  it('an empty payload still yields a complete, safe snapshot', () => {
    const snap = snapshotFromBlocks(blocksFromDetailed({} as DetailedHealth))
    expect(snap.model_loaded).toBe(false)
    expect(snap.health_score).toBe(0)
    expect(snap.health_status).toBe('unknown')
    expect(snap.diagnoses).toEqual([])
    expect(snap.path_latencies).toEqual([])
    expect(snap.cpu_percent).toBeNull()
    expect(snap.device).toBeNull()
  })

  it('non-numeric counters fall back instead of leaking NaN', () => {
    const snap = snapshotFromBlocks(
      blocksFromSse({ request_count: 'many', tokens_per_sec: 'fast' }),
    )
    expect(snap.request_count).toBe(0)
    expect(snap.tokens_per_sec).toBe(0)
  })

  it('non-array collections become empty arrays', () => {
    const snap = snapshotFromBlocks(blocksFromSse({ model_metrics: 'nope', recent_errors: 12 }))
    expect(snap.model_metrics).toEqual([])
    expect(snap.recent_errors).toEqual([])
  })
})
