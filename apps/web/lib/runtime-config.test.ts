import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as http from './http-client'
import {
  _resetRuntimeConfigCache,
  fetchRuntimeConfig,
  isFlagEnabled,
  type RuntimeConfig,
} from './runtime-config'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
}))

const apiGet = vi.mocked(http.apiGet)

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
      flags: DOC.flags.map((f) =>
        f.name === 'native_c_inference' ? { ...f, enabled: true } : f,
      ),
    })
    _resetRuntimeConfigCache()
    expect(await isFlagEnabled('native_c_inference')).toBe(true)
  })

  it('surfaces transport failures instead of inventing a default', async () => {
    apiGet.mockRejectedValue(new Error('boom'))
    await expect(fetchRuntimeConfig()).rejects.toThrow('boom')
  })
})
