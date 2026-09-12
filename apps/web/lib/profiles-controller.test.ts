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

import { profilesController } from './profiles-controller'

describe('profilesController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /profiles and returns data array', async () => {
    apiClient.apiGet.mockResolvedValue({ data: [{ id: 'p1', name: 'default' }] })
    const result = await profilesController.list()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/profiles')
    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('p1')
  })

  it('returns empty array when data is missing', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await profilesController.list()
    expect(result).toEqual([])
  })
})

describe('profilesController.get', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /profiles/{id} and returns profile', async () => {
    apiClient.apiGet.mockResolvedValue({ data: { id: 'p1', name: 'default', tier: 'small' } })
    const result = await profilesController.get('p1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/profiles/p1')
    expect(result.id).toBe('p1')
    expect(result.tier).toBe('small')
  })
})

describe('profilesController.apply', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /profiles/apply with profile_id', async () => {
    const mockResult = {
      profile: { id: 'p1', name: 'default' },
      live_settings: { temperature: 0.7 },
      requires_restart: [],
      active_profile_id: 'p1',
    }
    apiClient.apiPost.mockResolvedValue({ data: mockResult })
    const result = await profilesController.apply('p1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/profiles/apply', { profile_id: 'p1' })
    expect(result.active_profile_id).toBe('p1')
    expect(result.live_settings).toEqual({ temperature: 0.7 })
  })

  it('returns restart requirements when present', async () => {
    const mockResult = {
      profile: { id: 'p2', name: 'production' },
      live_settings: {},
      requires_restart: ['inference_pool_size', 'compute_threads'],
      active_profile_id: 'p2',
    }
    apiClient.apiPost.mockResolvedValue({ data: mockResult })
    const result = await profilesController.apply('p2')
    expect(result.requires_restart).toEqual(['inference_pool_size', 'compute_threads'])
  })
})

describe('profilesController.active', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /profiles/active and returns active profile', async () => {
    apiClient.apiGet.mockResolvedValue({ data: { active_profile_id: 'p1', profile: { id: 'p1', name: 'default' } } })
    const result = await profilesController.active()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/profiles/active')
    expect(result.active_profile_id).toBe('p1')
    expect(result.profile).not.toBeNull()
  })

  it('handles no active profile', async () => {
    apiClient.apiGet.mockResolvedValue({ data: { active_profile_id: null, profile: null } })
    const result = await profilesController.active()
    expect(result.active_profile_id).toBeNull()
    expect(result.profile).toBeNull()
  })
})

describe('profilesController.recommend', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /profiles/recommend and returns recommendation', async () => {
    apiClient.apiGet.mockResolvedValue({ data: { recommended_profile_id: 'p3', detected_ram_gb: 16, has_gpu: true } })
    const result = await profilesController.recommend()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/profiles/recommend')
    expect(result.recommended_profile_id).toBe('p3')
    expect(result.detected_ram_gb).toBe(16)
    expect(result.has_gpu).toBe(true)
  })

  it('handles no GPU detected', async () => {
    apiClient.apiGet.mockResolvedValue({ data: { recommended_profile_id: 'p1', detected_ram_gb: 8, has_gpu: false } })
    const result = await profilesController.recommend()
    expect(result.has_gpu).toBe(false)
  })
})
