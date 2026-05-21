/**
 * Load a .sou model with IndexedDB caching.
 *
 * On first call, fetches from URL and caches the ArrayBuffer.
 * On subsequent calls, serves from cache — no network request.
 */

import { SoulNetWebGPU, type SoulNetConfig } from './engine'
import { inferArch } from './weights'
import { getCached, setCached } from './cache'

/** Fetch a .sou buffer (from cache or network). */
export async function fetchSou(url: string): Promise<ArrayBuffer> {
  const cached = await getCached(url)
  if (cached) return cached
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`)
  const buffer = await resp.arrayBuffer()
  setCached(url, buffer).catch(() => {})
  return buffer
}

/** Load a model by URL, auto-detecting architecture. */
export async function cachedLoadAuto(
  engine: SoulNetWebGPU,
  url: string,
  overrides?: Partial<SoulNetConfig>,
): Promise<void> {
  const buffer = await fetchSou(url)
  const arch = inferArch(buffer)
  const config: SoulNetConfig = { ...arch, ...overrides }
  await engine.load(buffer, config)
}
