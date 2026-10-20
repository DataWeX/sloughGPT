import { describe, it, expect, beforeEach, vi } from 'vitest'
import { WeightCache } from './cache'
import { installFakeIndexedDB } from './__test-helper'

describe('WeightCache', () => {
  let idb: ReturnType<typeof installFakeIndexedDB>
  let cache: WeightCache

  beforeEach(() => {
    idb = installFakeIndexedDB()
    cache = new WeightCache()
  })

  it('get returns null for missing key', async () => {
    const result = await cache.get('https://models.example/missing.soul')
    expect(result).toBeNull()
  })

  it('put then get returns the same buffer', async () => {
    const buf = new ArrayBuffer(16)
    new Uint8Array(buf).set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    await cache.put('https://models.example/test.soul', buf)
    const result = await cache.get('https://models.example/test.soul')
    expect(result).not.toBeNull()
    expect(result!.byteLength).toBe(16)
    expect(new Uint8Array(result!)).toEqual(new Uint8Array(buf))
  })

  it('clear empties the store', async () => {
    const buf = new ArrayBuffer(8)
    await cache.put('https://models.example/a.soul', buf)
    await cache.put('https://models.example/b.soul', buf)
    expect(await cache.get('https://models.example/a.soul')).not.toBeNull()
    await cache.clear()
    expect(await cache.get('https://models.example/a.soul')).toBeNull()
    expect(await cache.get('https://models.example/b.soul')).toBeNull()
  })

  it('different URLs are independent', async () => {
    const bufA = new ArrayBuffer(4)
    new DataView(bufA).setUint32(0, 0xAAAAAAAA)
    const bufB = new ArrayBuffer(4)
    new DataView(bufB).setUint32(0, 0xBBBBBBBB)
    await cache.put('https://models.example/a.soul', bufA)
    await cache.put('https://models.example/b.soul', bufB)
    const a = await cache.get('https://models.example/a.soul')
    const b = await cache.get('https://models.example/b.soul')
    expect(new DataView(a!).getUint32(0)).toBe(0xAAAAAAAA)
    expect(new DataView(b!).getUint32(0)).toBe(0xBBBBBBBB)
  })

  it('put overwrites existing entry', async () => {
    const buf1 = new ArrayBuffer(4)
    new DataView(buf1).setUint32(0, 111)
    const buf2 = new ArrayBuffer(4)
    new DataView(buf2).setUint32(0, 222)
    await cache.put('https://models.example/x.soul', buf1)
    await cache.put('https://models.example/x.soul', buf2)
    const result = await cache.get('https://models.example/x.soul')
    expect(new DataView(result!).getUint32(0)).toBe(222)
  })

  it('works when IndexedDB is unavailable', async () => {
    vi.stubGlobal('indexedDB', undefined)
    const fallback = new WeightCache()
    const buf = new ArrayBuffer(4)
    const result = await fallback.get('https://models.example/x.soul')
    expect(result).toBeNull()
    await expect(fallback.put('https://models.example/x.soul', buf)).resolves.not.toThrow()
    await expect(fallback.clear()).resolves.not.toThrow()
  })
})
