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
    apiGet.mockResolvedValue({ data: { models: [{ model_id: 'gpt2', status: 'loaded' }] } })
    const result = await registryController.list()
    expect(result).toEqual([{ model_id: 'gpt2', status: 'loaded' }])
  })

  it('list handles flat array', async () => {
    apiGet.mockResolvedValue([{ model_id: 'qwen', status: 'registered' }])
    const result = await registryController.list()
    expect(result).toEqual([{ model_id: 'qwen', status: 'registered' }])
  })

  it('stats unwraps data envelope', async () => {
    apiGet.mockResolvedValue({ data: { total_models: 3, loaded_models: 1, failed_models: 0, circuit_breaker_open: false } })
    const result = await registryController.stats()
    expect(result).toEqual({ total_models: 3, loaded_models: 1, failed_models: 0, circuit_breaker_open: false })
  })

  it('stats handles flat response', async () => {
    apiGet.mockResolvedValue({ total_models: 1, loaded_models: 0, failed_models: 1, circuit_breaker_open: true })
    const result = await registryController.stats()
    expect(result).toEqual({ total_models: 1, loaded_models: 0, failed_models: 1, circuit_breaker_open: true })
  })

  it('best unwraps data envelope', async () => {
    apiGet.mockResolvedValue({ data: { model_id: 'gpt2', score: 0.95 } })
    const result = await registryController.best()
    expect(result).toEqual({ model_id: 'gpt2', score: 0.95 })
  })

  it('best returns null on empty', async () => {
    apiGet.mockResolvedValue(null)
    const result = await registryController.best()
    expect(result).toBeNull()
  })
})
