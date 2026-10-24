'use client'

// Protocol state — the frontend's SyncAgent analogue (learned from
// Towns/River: components read protocol state from one store; the network is
// a seam under it). Descriptor-driven semantics:
//   * idempotent capabilities cache — a re-read with unchanged params+version
//     is served from the store, not the wire;
//   * the descriptor version is part of the key — a contract bump invalidates
//     every read taken under the old one;
//   * http-client stays the transport seam (AGENTS.md http-client rule);
//     the store never touches fetch().

import { useStore } from 'zustand'
import { createStore } from 'zustand/vanilla'

import { apiGet } from '@/lib/http-client'

import { CONTRACTS, type ContractName } from './contracts.gen'
import type { EndpointParams, ReadSlice } from './types'

export const IDLE_SLICE: ReadSlice = { status: 'idle', updatedAt: 0 }

/** Stable cache key: capability @ descriptor version # sorted params. */
export function readKey(name: ContractName, params?: EndpointParams): string {
  const version = CONTRACTS[name]?.version ?? 'unregistered'
  const suffix = params
    ? Object.keys(params)
        .sort()
        .map((k) => `${k}=${String(params[k])}`)
        .join('&')
    : ''
  return `${name}@${version}#${suffix}`
}

export interface FetchOptions {
  /** Bypass the idempotent cache — used by explicit refetch. */
  force?: boolean
}

interface ProtocolStore {
  reads: Record<string, ReadSlice>
  fetchRead: (name: ContractName, params?: EndpointParams, opts?: FetchOptions) => Promise<void>
  /** Drop cached reads — all of them, or only one capability's. */
  invalidate: (name?: ContractName) => void
}

export const protocolStore = createStore<ProtocolStore>((set, get) => ({
  reads: {},

  fetchRead: async (name, params, opts) => {
    const descriptor = CONTRACTS[name]
    const key = readKey(name, params)

    if (!descriptor) {
      set((s) => ({
        reads: {
          ...s.reads,
          [key]: { status: 'error', error: `unknown capability: ${name}`, updatedAt: Date.now() },
        },
      }))
      return
    }

    const prev = get().reads[key]
    if (!opts?.force && prev) {
      if (prev.status === 'loading') return // in-flight: one request per key
      if (descriptor.idempotent && prev.status === 'success') return // cached read
    }

    set((s) => ({
      reads: {
        ...s.reads,
        [key]: { status: 'loading', data: prev?.data, updatedAt: prev?.updatedAt ?? 0 },
      },
    }))

    try {
      if (descriptor.method !== 'GET') {
        throw new Error(
          `${descriptor.method} crossings are not readable through useEndpoint (GET reads only)`,
        )
      }
      const qs = params
        ? Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)]))
        : undefined
      const data = await apiGet<unknown>(descriptor.path, qs)
      set((s) => ({
        reads: { ...s.reads, [key]: { status: 'success', data, updatedAt: Date.now() } },
      }))
    } catch (err) {
      set((s) => ({
        reads: {
          ...s.reads,
          [key]: {
            status: 'error',
            data: prev?.data, // keep last good data while surfacing the error
            error: err instanceof Error ? err.message : String(err),
            updatedAt: prev?.updatedAt ?? 0,
          },
        },
      }))
    }
  },

  invalidate: (name) => {
    set((s) => {
      if (!name) return { reads: {} }
      const prefix = `${name}@`
      return {
        reads: Object.fromEntries(Object.entries(s.reads).filter(([k]) => !k.startsWith(prefix))),
      }
    })
  },
}))

export const useProtocolStore = Object.assign(
  <T>(selector: (state: ProtocolStore) => T): T => useStore(protocolStore, selector),
  { getState: protocolStore.getState },
)
