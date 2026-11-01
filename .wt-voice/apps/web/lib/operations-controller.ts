'use client'

import { apiGet, apiPost } from './http-client'

export interface Operation {
  id: string
  type: string
  label: string
  status: string
  created_at: number
  started_at: number | null
  finished_at: number | null
  elapsed_s: number
  error: string | null
  meta: Record<string, unknown>
}

export interface OperationsResponse {
  operations: Operation[]
  counts: Record<string, number>
}

export interface CancelResponse {
  cancelled: string[]
  count: number
}

export const operationsController = {
  list(type?: string): Promise<OperationsResponse> {
    const qs = type ? `?type=${type}` : ''
    return apiGet<OperationsResponse>(`/operations${qs}`)
  },

  cancel(opId: string): Promise<Operation> {
    return apiPost<Operation>(`/cancel/${opId}`)
  },

  cancelAll(type?: string): Promise<CancelResponse> {
    const qs = type ? `?type=${type}` : ''
    return apiPost<CancelResponse>(`/cancel-all${qs}`)
  },

  purge(maxAgeS?: number): Promise<{ purged: number }> {
    const qs = maxAgeS != null ? `?max_age_s=${maxAgeS}` : ''
    return apiPost<{ purged: number }>(`/operations/purge${qs}`)
  },
}
