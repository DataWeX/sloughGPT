import { describe, it, expect, vi, beforeEach } from 'vitest'
import { operationsController } from './operations-controller'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

import { apiGet, apiPost } from '@/lib/http-client'
const mockGet = vi.mocked(apiGet)
const mockPost = vi.mocked(apiPost)

beforeEach(() => vi.clearAllMocks())

describe('operationsController', () => {
  it('list without type', async () => {
    mockGet.mockResolvedValueOnce({ operations: [], counts: {} })
    const r = await operationsController.list()
    expect(mockGet).toHaveBeenCalledWith('/operations')
    expect(r.operations).toEqual([])
  })

  it('list with type', async () => {
    mockGet.mockResolvedValueOnce({ operations: [], counts: {} })
    await operationsController.list('training')
    expect(mockGet).toHaveBeenCalledWith('/operations?type=training')
  })

  it('cancel posts to /cancel/{id}', async () => {
    mockPost.mockResolvedValueOnce({ id: 'op1', status: 'cancelled' })
    await operationsController.cancel('op1')
    expect(mockPost).toHaveBeenCalledWith('/cancel/op1')
  })

  it('cancelAll without type', async () => {
    mockPost.mockResolvedValueOnce({ cancelled: [], count: 0 })
    await operationsController.cancelAll()
    expect(mockPost).toHaveBeenCalledWith('/cancel-all')
  })

  it('cancelAll with type', async () => {
    mockPost.mockResolvedValueOnce({ cancelled: ['x'], count: 1 })
    await operationsController.cancelAll('inference')
    expect(mockPost).toHaveBeenCalledWith('/cancel-all?type=inference')
  })

  it('purge without maxAge', async () => {
    mockPost.mockResolvedValueOnce({ purged: 3 })
    const r = await operationsController.purge()
    expect(mockPost).toHaveBeenCalledWith('/operations/purge')
    expect(r.purged).toBe(3)
  })

  it('purge with maxAge', async () => {
    mockPost.mockResolvedValueOnce({ purged: 1 })
    await operationsController.purge(600)
    expect(mockPost).toHaveBeenCalledWith('/operations/purge?max_age_s=600')
  })
})
