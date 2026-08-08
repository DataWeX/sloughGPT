import { describe, it, expect, afterEach, vi } from 'vitest'
import { getCached, setCached, hasCached, clearCache } from './cache'
import { installFakeIndexedDB } from './__test-helper'

const URL = 'https://models.example/friendly.sou'
const makeBuf = () => new Uint8Array([1, 2, 3, 4]).buffer

describe('soulnet-webgpu cache', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('opens the correct DB name and version', async () => {
    const { open } = installFakeIndexedDB()
    await getCached(URL)
    expect(open).toHaveBeenCalledWith('soulnet-cache', 1)
  })

  it('getCached returns null on miss', async () => {
    installFakeIndexedDB()
    expect(await getCached(URL)).toBeNull()
  })

  it('setCached then getCached roundtrips the ArrayBuffer', async () => {
    const { store } = installFakeIndexedDB()
    const data = makeBuf()
    await setCached(URL, data)
    expect(store.get(URL)).toBe(data)
    const got = await getCached(URL)
    expect(got).toBe(data)
    expect(Array.from(new Uint8Array(got as ArrayBuffer))).toEqual([1, 2, 3, 4])
  })

  it('hasCached reflects store contents', async () => {
    installFakeIndexedDB()
    expect(await hasCached(URL)).toBe(false)
    await setCached(URL, makeBuf())
    expect(await hasCached(URL)).toBe(true)
  })

  it('keys are isolated per URL', async () => {
    installFakeIndexedDB()
    await setCached(URL, makeBuf())
    expect(await getCached(URL)).not.toBeNull()
    expect(await getCached('https://other.example/x.sou')).toBeNull()
  })

  it('clearCache empties the store', async () => {
    const { store } = installFakeIndexedDB()
    await setCached(URL, makeBuf())
    await setCached('https://models.example/other.sou', makeBuf())
    await clearCache()
    expect(store.size).toBe(0)
    expect(await getCached(URL)).toBeNull()
  })
})
