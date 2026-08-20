import { describe, it, expect, vi, beforeEach } from 'vitest'
import { registryController } from './registry-controller'
import * as http from './http-client'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
}))

const apiGet = vi.mocked(http.apiGet)

beforeEach(() => { vi.clearAllMocks() })

describe('registryController', () => {
  it('list unwraps nested data.models', async () => {
    apiGet.mockResolvedValue({ models: [{ model_id: 'gpt2', status: 'ready' }] })
    const result = await registryController.list()
    expect(result).toEqual([{ model_id: 'gpt2', status: 'ready' }])
  })

  it('list handles flat array', async () => {
    apiGet.mockResolvedValue([{ model_id: 'qwen', status: 'registered' }])
    const result = await registryController.list()
    expect(result).toEqual([{ model_id: 'qwen', status: 'registered' }])
  })

  it('stats maps backend fields to frontend shape', async () => {
    apiGet.mockResolvedValue({ models_registered: 3, models_loaded: 1, healthy: true, degraded: false, has_errors: false, default_model: 'gpt2' })
    const result = await registryController.stats()
    expect(result).toEqual({ total_models: 3, loaded_models: 1, failed_models: 0, circuit_breaker_open: false })
  })

  it('stats maps errors and degraded', async () => {
    apiGet.mockResolvedValue({ models_registered: 1, models_loaded: 0, healthy: false, degraded: true, has_errors: true })
    const result = await registryController.stats()
    expect(result).toEqual({ total_models: 1, loaded_models: 0, failed_models: 1, circuit_breaker_open: true })
  })

  it('best returns the data as-is', async () => {
    apiGet.mockResolvedValue({ default_model: 'gpt2', models_loaded: 1 })
    const result = await registryController.best()
    expect(result).toEqual({ default_model: 'gpt2', models_loaded: 1 })
  })

  it('best returns null on empty', async () => {
    apiGet.mockResolvedValue(null)
    const result = await registryController.best()
    expect(result).toBeNull()
  })

  it('list calls correct endpoint', async () => {
    apiGet.mockResolvedValue([])
    await registryController.list()
    expect(apiGet).toHaveBeenCalledWith('/registry/models')
  })

  it('stats calls correct endpoint', async () => {
    apiGet.mockResolvedValue({ models_registered: 0, models_loaded: 0, healthy: true, degraded: false, has_errors: false })
    await registryController.stats()
    expect(apiGet).toHaveBeenCalledWith('/registry/stats')
  })
})
