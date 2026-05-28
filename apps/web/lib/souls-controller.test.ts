import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { soulsController } from './souls-controller'

describe('soulsController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /souls and returns souls', async () => {
    apiClient.apiGet.mockResolvedValue({ souls: [{ name: 'friendly', description: 'Friendly soul', traits: ['warm'] }] })

    const result = await soulsController.list()
    expect(result.souls).toHaveLength(1)
    expect(result.souls[0].name).toBe('friendly')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/souls')
  })
})

describe('soulsController.getCurrent', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /souls/current', async () => {
    apiClient.apiGet.mockResolvedValue({ name: 'friendly', description: '', traits: ['warm'], personality: { warmth: 0.8 } })

    const result = await soulsController.getCurrent()
    expect(result?.name).toBe('friendly')
    expect(result?.personality?.warmth).toBe(0.8)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/souls/current')
  })

  it('returns null on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('404'))
    expect(await soulsController.getCurrent()).toBeNull()
  })
})

describe('soulsController.switch', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /souls/switch with name', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok' })

    await soulsController.switch('friendly')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/souls/switch', { name: 'friendly' })
  })

  it('includes checkpoint_name when provided', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok' })

    await soulsController.switch('friendly', 'v2')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/souls/switch', { name: 'friendly', checkpoint_name: 'v2' })
  })
})

describe('soulsController.listCheckpoints', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /auto-train/checkpoints', async () => {
    apiClient.apiGet.mockResolvedValue({ checkpoints: [{ name: 'v1', soul: 'friendly' }] })

    const result = await soulsController.listCheckpoints()
    expect(result.checkpoints).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/auto-train/checkpoints')
  })
})

describe('soulsController.loadCheckpoint', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /auto-train/checkpoints/{name}/load', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'loaded', name: 'v1', soul: 'friendly' })

    const result = await soulsController.loadCheckpoint('v1')
    expect(result.status).toBe('loaded')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/auto-train/checkpoints/v1/load')
  })
})
