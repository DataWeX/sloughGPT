'use client'

// The frontend's useTimeline analogue (Towns/River): a thin hook over the
// protocol store — components name a capability, never a URL.

import { useEffect, useRef } from 'react'

import type { ContractName } from './contracts.gen'
import { IDLE_SLICE, readKey, useProtocolStore } from './store'
import type { EndpointParams, ReadStatus } from './types'

export interface EndpointResult<T> {
  data: T | undefined
  status: ReadStatus
  error: string | undefined
  /** Re-read past the idempotent cache (force + transport bypass). */
  refetch: () => void
}

export function useEndpoint<T>(name: ContractName, params?: EndpointParams): EndpointResult<T> {
  const key = readKey(name, params)
  const slice = useProtocolStore((s) => s.reads[key] ?? IDLE_SLICE)
  const fetchRead = useProtocolStore((s) => s.fetchRead)

  // Params arrive as object literals each render; the stable key IS the
  // identity. Keep the latest literal in a ref so refetch sees fresh values
  // without re-running the effect.
  const paramsRef = useRef(params)
  paramsRef.current = params

  useEffect(() => {
    void fetchRead(name, paramsRef.current)
  }, [fetchRead, key, name, paramsRef])

  return {
    data: slice.data as T | undefined,
    status: slice.status,
    error: slice.error,
    refetch: () => {
      void fetchRead(name, paramsRef.current, { force: true })
    },
  }
}
