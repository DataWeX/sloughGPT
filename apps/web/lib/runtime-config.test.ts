import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { _globalInterceptors } from '../../lib/state'
import type { RequestConfig, ResponseEnvelope } from '../../lib/types'
import * as http from './http-client'
import { notify } from './notify'
import {
  CONFIG_VERSION_HEADER,
  _resetRuntimeConfigCache,
  _resetRuntimeConfigGuards,
  attachConfigVersion,
  fetchRuntimeConfig,
  handleConfigDrift,
  installRuntimeConfigGuards,
  isFlagEnabled,
  type RuntimeConfig,
} from './runtime-config'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
}))

vi.mock('./notify', () => ({
  notify: { toast: vi.fn(), banner: vi.fn() },
}))

const apiGet = vi.mocked(http.apiGet)
const banner = vi.mocked(notify.banner)

const DOC: RuntimeConfig = {
  config_version: 'a'.repeat(64),
  flags: [
    {
      name: 'slonet_provider',
      status: 'enabled',
      enabled: true,
      version: '1.0.0',
      since: '2026-11-11',
      description: 'Sole inference engine',
    },
    {
      name: 'native_c_inference',
      status: 'disabled',
      enabled: false,
      version: '1.1.0',
      since: '2026-11-11',
      description: 'C transformer forward pass',
    },
  ],
}

beforeEach(() => {
  vi.clearAllMocks()
  _resetRuntimeConfigCache()
})

afterEach(() => {
  vi.useRealTimers()
  _globalInterceptors.request.clear()
  _globalInterceptors.response.clear()
  _resetRuntimeConfigGuards()
})

describe('runtime-config', () => {
  it('fetchRuntimeConfig returns the bare document via apiGet', async () => {
    apiGet.mockResolvedValue(DOC)

    const cfg = await fetchRuntimeConfig()

    // Raw-wire facet: no {status, data} envelope — apiGet passes it through.
    expect(apiGet).toHaveBeenCalledWith('/runtime-config')
    expect(cfg.config_version).toBe('a'.repeat(64))
    expect(cfg.flags).toHaveLength(2)
  })

  it('serves from the in-memory TTL within 30s of the header', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(0)
    apiGet.mockResolvedValue(DOC)

    await fetchRuntimeConfig()
    vi.setSystemTime(29_000)
    await fetchRuntimeConfig()
    expect(apiGet).toHaveBeenCalledTimes(1) // still inside the 30s window

    vi.setSystemTime(31_000)
    await fetchRuntimeConfig()
    expect(apiGet).toHaveBeenCalledTimes(2) // window elapsed → refetch
  })

  it('isFlagEnabled reads flag state from the response, not from code', async () => {
    apiGet.mockResolvedValue(DOC)

    expect(await isFlagEnabled('slonet_provider')).toBe(true)
    expect(await isFlagEnabled('native_c_inference')).toBe(false)
    // A flag the server never declared must fail safe (matches FeatureFlags).
    expect(await isFlagEnabled('nonexistent_flag')).toBe(false)

    // Flip the server's answer — the consumer follows it, no local constant.
    apiGet.mockResolvedValue({
      ...DOC,
      config_version: 'b'.repeat(64),
      flags: DOC.flags.map((f) => (f.name === 'native_c_inference' ? { ...f, enabled: true } : f)),
    })
    _resetRuntimeConfigCache()
    expect(await isFlagEnabled('native_c_inference')).toBe(true)
  })

  it('surfaces transport failures instead of inventing a default', async () => {
    apiGet.mockRejectedValue(new Error('boom'))
    await expect(fetchRuntimeConfig()).rejects.toThrow('boom')
  })
})

// ── Drift guards (card 20261108_222308) ─────────────────────────────────────

const postConfig = (): RequestConfig => ({
  method: 'POST',
  url: '/some/action',
  fullUrl: '/some/action',
  headers: {},
  body: { n: 1 },
  correlationId: 'corr-1',
  startTime: 0,
  opts: {},
})

const driftEnvelope = (status: number, body: unknown): ResponseEnvelope => ({
  ok: false,
  status,
  statusText: status === 412 ? 'Precondition Failed' : 'Error',
  headers: new Headers(),
  text: JSON.stringify(body),
  config: postConfig(),
})

const SENTENCE =
  'This page and the server disagree about the active configuration, ' +
  'so nothing was sent. Reload the page to continue.'

describe('config-drift guards', () => {
  it('presents the held config_version on unsafe requests', async () => {
    apiGet.mockResolvedValue(DOC)

    const cfg = await attachConfigVersion(postConfig())

    expect(cfg.headers[CONFIG_VERSION_HEADER]).toBe('a'.repeat(64))
    expect(apiGet).toHaveBeenCalledWith('/runtime-config')
  })

  it('never presents on reads — GETs are the recovery path', async () => {
    apiGet.mockResolvedValue(DOC)

    const cfg = await attachConfigVersion({ ...postConfig(), method: 'GET' })

    expect(cfg.headers[CONFIG_VERSION_HEADER]).toBeUndefined()
    expect(apiGet).not.toHaveBeenCalled()
  })

  it('makes no claim when the document cannot be fetched', async () => {
    apiGet.mockRejectedValue(new Error('offline'))

    const cfg = await attachConfigVersion(postConfig())

    expect(cfg.headers[CONFIG_VERSION_HEADER]).toBeUndefined()
  })

  it('never overwrites an explicitly presented claim', async () => {
    const cfg = postConfig()
    cfg.headers[CONFIG_VERSION_HEADER] = 'client-pinned'

    const out = await attachConfigVersion(cfg)

    expect(out.headers[CONFIG_VERSION_HEADER]).toBe('client-pinned')
    expect(apiGet).not.toHaveBeenCalled()
  })

  it('on 412 E_CONFIG_DRIFT: drops the stale doc, raises the reload banner, lets the refusal propagate', async () => {
    apiGet.mockResolvedValue(DOC)
    await fetchRuntimeConfig() // hold N-1
    expect(apiGet).toHaveBeenCalledTimes(1)

    const env = driftEnvelope(412, { error: SENTENCE, code: 'E_CONFIG_DRIFT' })
    const out = handleConfigDrift(env)

    expect(out).toBe(env) // envelope unchanged → ApiError still thrown to caller
    expect(banner).toHaveBeenCalledTimes(1)
    expect(banner).toHaveBeenCalledWith(
      expect.objectContaining({
        tone: 'destructive',
        key: 'config-drift',
        title: 'Configuration out of date',
        message: SENTENCE,
        action: expect.objectContaining({ label: 'Reload the page' }),
      }),
    )

    // Stale calibration dropped: the next read goes back to the wire.
    apiGet.mockClear()
    await fetchRuntimeConfig()
    expect(apiGet).toHaveBeenCalledTimes(1)
  })

  it('ignores 412s with another code and non-412 statuses', () => {
    handleConfigDrift(driftEnvelope(412, { error: 'busy', code: 'E_CONFLICT' }))
    handleConfigDrift(driftEnvelope(500, { error: SENTENCE, code: 'E_CONFIG_DRIFT' }))
    handleConfigDrift(driftEnvelope(412, 'not json at all'))
    expect(banner).not.toHaveBeenCalled()
  })

  it('installs once and both guards work through the global chain', async () => {
    installRuntimeConfigGuards()
    installRuntimeConfigGuards()
    expect(_globalInterceptors.request.size).toBe(1)
    expect(_globalInterceptors.response.size).toBe(1)

    apiGet.mockResolvedValue(DOC)
    const ran = await _globalInterceptors.request.run(postConfig())
    expect(ran.headers[CONFIG_VERSION_HEADER]).toBe('a'.repeat(64))

    const env = await _globalInterceptors.response.run(
      driftEnvelope(412, { error: SENTENCE, code: 'E_CONFIG_DRIFT' }),
    )
    expect(env.status).toBe(412)
    expect(banner).toHaveBeenCalledTimes(1)
  })
})
