import { beforeEach, describe, expect, it, vi } from 'vitest'

import { apiGet } from '@/lib/http-client'

import { protocolStore, readKey } from './store'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
}))

const apiGetMock = vi.mocked(apiGet)

function slice(key: string) {
  return protocolStore.getState().reads[key]
}

beforeEach(() => {
  protocolStore.getState().invalidate()
  apiGetMock.mockReset()
})

describe('protocol store — descriptor-driven reads', () => {
  it('resolves the path from the descriptor, not from call-site strings', async () => {
    apiGetMock.mockResolvedValue({ history: [{ step: 0 }] })

    await protocolStore
      .getState()
      .fetchRead('consciousness.personality_history')

    expect(apiGetMock).toHaveBeenCalledWith('/consciousness/personality/history', undefined)
    const key = readKey('consciousness.personality_history')
    expect(slice(key)).toMatchObject({ status: 'success', data: { history: [{ step: 0 }] } })
  })

  it('caches idempotent reads until invalidated', async () => {
    apiGetMock.mockResolvedValue({ history: [] })

    await protocolStore.getState().fetchRead('consciousness.personality_history')
    await protocolStore.getState().fetchRead('consciousness.personality_history')
    expect(apiGetMock).toHaveBeenCalledTimes(1)

    protocolStore.getState().invalidate('consciousness.personality_history')
    await protocolStore.getState().fetchRead('consciousness.personality_history')
    expect(apiGetMock).toHaveBeenCalledTimes(2)
  })

  it('force bypasses the cache for explicit refetch', async () => {
    apiGetMock.mockResolvedValue({ history: [] })

    await protocolStore.getState().fetchRead('consciousness.personality_history')
    await protocolStore
      .getState()
      .fetchRead('consciousness.personality_history', undefined, { force: true })
    expect(apiGetMock).toHaveBeenCalledTimes(2)
  })

  it('dedupes concurrent reads of the same key', async () => {
    apiGetMock.mockReturnValue(new Promise(() => {})) // never settles

    void protocolStore.getState().fetchRead('consciousness.personality_history')
    void protocolStore.getState().fetchRead('consciousness.personality_history')
    await Promise.resolve()

    expect(apiGetMock).toHaveBeenCalledTimes(1)
    expect(slice(readKey('consciousness.personality_history'))?.status).toBe('loading')
  })

  it('key order is stable — params are sorted, not positional', () => {
    const a = readKey('contracts.list', { prefix: '/x', extra: '1' })
    const b = readKey('contracts.list', { extra: '1', prefix: '/x' })
    expect(a).toBe(b)
  })

  it('the descriptor version is part of the key (contract bump ⇒ cold read)', () => {
    const key = readKey('consciousness.personality_history')
    expect(key).toBe('consciousness.personality_history@1#')
  })

  it('surfaces failures as an error slice without throwing', async () => {
    apiGetMock.mockRejectedValue(new Error('network down'))

    await protocolStore.getState().fetchRead('consciousness.personality_history')

    const key = readKey('consciousness.personality_history')
    expect(slice(key)?.status).toBe('error')
    expect(slice(key)?.error).toBe('network down')
  })

  it('rejects unknown capabilities with a named error', async () => {
    await protocolStore.getState().fetchRead('not.a.capability' as never)

    const key = readKey('not.a.capability' as never)
    expect(slice(key)?.status).toBe('error')
    expect(slice(key)?.error).toContain('unknown capability')
  })
})
