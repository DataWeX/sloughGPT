import { describe, expect, it, vi, beforeEach } from 'vitest'

const mockApiGet = vi.hoisted(() => vi.fn())

vi.mock('@/lib/http-client', () => ({ apiGet: mockApiGet }))

import { bootstrapController, normalizeHfModels } from './bootstrap-controller'

describe('normalizeHfModels', () => {
  it('passes a ModelInfo array through', () => {
    const list = [{ id: 'a/b', name: 'B' }]
    expect(normalizeHfModels(list)).toEqual(list)
  })

  it('unwraps the { models: [...] } envelope variant', () => {
    expect(normalizeHfModels({ models: [{ id: 'x', name: 'X' }] })).toEqual([
      { id: 'x', name: 'X' },
    ])
  })

  it('expands bare model-id strings', () => {
    expect(normalizeHfModels(['a/b'])).toEqual([{ id: 'a/b', name: 'a/b', type: 'huggingface' }])
  })

  it('returns [] for null/undefined', () => {
    expect(normalizeHfModels(null)).toEqual([])
    expect(normalizeHfModels(undefined)).toEqual([])
  })
})

describe('bootstrapController.get', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('normalizes the models member in place', async () => {
    mockApiGet.mockResolvedValue({
      members: { models: { models: ['a/b', { id: 'c', name: 'C' }] } },
      errors: {},
    })
    const payload = await bootstrapController.get()
    expect(payload.members.models).toEqual([
      { id: 'a/b', name: 'a/b', type: 'huggingface' },
      { id: 'c', name: 'C' },
    ])
  })

  it('leaves a null models member as null (not_ready/degraded)', async () => {
    mockApiGet.mockResolvedValue({ members: { models: null }, errors: { models: 'not_ready' } })
    const payload = await bootstrapController.get()
    expect(payload.members.models).toBeNull()
    expect(payload.errors.models).toBe('not_ready')
  })
})
