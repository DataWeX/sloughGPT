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

import { consciousnessController } from './consciousness-controller'

describe('consciousnessController.getStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /consciousness/status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'active', uptime: 3600, episodes: 42, beliefs_count: 10, qualia_count: 5 })
    const result = await consciousnessController.getStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/consciousness/status')
    expect(result.status).toBe('active')
    expect(result.episodes).toBe(42)
  })
})

describe('consciousnessController.getSelfModel', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /consciousness/self-model', async () => {
    apiClient.apiGet.mockResolvedValue({ traits: { curiosity: 0.8 }, beliefs: ['b1'], goals: ['g1'], memories: ['m1'] })
    const result = await consciousnessController.getSelfModel()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/consciousness/self-model')
    expect(result.traits.curiosity).toBe(0.8)
  })
})

describe('consciousnessController.reflect', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /consciousness/reflect with prompt', async () => {
    apiClient.apiPost.mockResolvedValue({ reflection: 'I am learning...' })
    const result = await consciousnessController.reflect()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/consciousness/reflect')
    expect(result.reflection).toBe('I am learning...')
  })
})

describe('consciousnessController.getTrainingStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /consciousness/train/status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'running', epoch: 3, loss: 0.42, episodes_trained: 100 })
    const result = await consciousnessController.getTrainingStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/consciousness/train/status')
    expect(result.epoch).toBe(3)
    expect(result.loss).toBe(0.42)
  })
})

describe('consciousnessController.startTraining', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /consciousness/train/start', async () => {
    apiClient.apiPost.mockResolvedValue({ started: true })
    const result = await consciousnessController.startTraining()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/consciousness/train/start', {})
    expect(result.started).toBe(true)
  })

  it('passes config to training start', async () => {
    apiClient.apiPost.mockResolvedValue({ started: true })
    await consciousnessController.startTraining({ epochs: 10, lr: 0.001 })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/consciousness/train/start', { epochs: 10, lr: 0.001 })
  })
})

describe('consciousnessController.getPersonality', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /consciousness/personality', async () => {
    apiClient.apiGet.mockResolvedValue({ traits: { openness: 0.9 }, description: 'Curious', name: 'Explorer' })
    const result = await consciousnessController.getPersonality()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/consciousness/personality')
    expect(result.name).toBe('Explorer')
  })
})

describe('consciousnessController.updatePersonality', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PATCHes /consciousness/personality with traits', async () => {
    apiClient.apiPatch.mockResolvedValue({ updated: true })
    const result = await consciousnessController.updatePersonality({ openness: 0.95 })
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/consciousness/personality', { traits: { openness: 0.95 } })
    expect(result.updated).toBe(true)
  })
})

describe('consciousnessController.listPersonas', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /consciousness/personas', async () => {
    apiClient.apiGet.mockResolvedValue({ personas: [{ id: 'p1', name: 'Explorer', active: true }] })
    const result = await consciousnessController.listPersonas()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/consciousness/personas')
    expect(result.personas).toHaveLength(1)
  })
})

describe('consciousnessController.savePersona', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /consciousness/personas/save', async () => {
    apiClient.apiPost.mockResolvedValue({ id: 'p2' })
    const result = await consciousnessController.savePersona({ name: 'Thinker', personality: { traits: {}, description: '', name: '' }, active: false })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/consciousness/personas/save', expect.objectContaining({ name: 'Thinker' }))
    expect(result.id).toBe('p2')
  })
})

describe('consciousnessController.deletePersona', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /consciousness/personas/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue({ deleted: true })
    const result = await consciousnessController.deletePersona('p1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/consciousness/personas/p1')
    expect(result.deleted).toBe(true)
  })
})

describe('consciousnessController.backup', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /consciousness/backup', async () => {
    apiClient.apiPost.mockResolvedValue({ backup_id: 'b1', size_bytes: 1024, created_at: '2026-01-01' })
    const result = await consciousnessController.backup()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/consciousness/backup')
    expect(result.backup_id).toBe('b1')
  })
})

describe('consciousnessController.healthCheck', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /consciousness/health', async () => {
    apiClient.apiGet.mockResolvedValue({ health_score: 85, enabled: true, level: 2, episodes: 10, avg_growth: 0.05, positive_ratio: 0.8, qualia: {}, last_reflection: 'test', diagnostics: [] })
    const result = await consciousnessController.healthCheck()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/consciousness/health')
    expect(result.health_score).toBe(85)
  })
})
