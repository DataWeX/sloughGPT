/**
 * Rate Limit controller — status and check.
 */
import { apiGet } from '@/lib/http-client'

export interface RateLimitStatus {
  requests_per_minute: number
  burst_size: number
  enabled: boolean
}

export interface RateLimitCheck {
  allowed: boolean
  wait_time: number
}

export const rateLimitController = {
  async getStatus(): Promise<RateLimitStatus> {
    return apiGet<RateLimitStatus>('/rate-limit/status')
  },

  async check(): Promise<RateLimitCheck> {
    return apiGet<RateLimitCheck>('/rate-limit/check')
  },
}
