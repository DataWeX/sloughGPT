/**
 * System Controller — system metrics, info, disk, and detailed health.
 *
 * Usage:
 *   import { systemController } from '@/lib/system-controller'
 *   const metrics = await systemController.getMetrics()
 */

import { apiGet } from './http-client'

export interface SystemMetrics {
  cpu_percent: number
  memory_percent: number
  memory_used_gb: number
  memory_total_gb: number
}

export interface SystemInfo {
  platform: string
  platform_release: string
  platform_version: string
  architecture: string
  processor: string
  cpu_count: number
}

export interface DiskUsage {
  total_gb: number
  used_gb: number
  free_gb: number
  percent: number
}

export interface GPUInfo {
  backend: string
  device_type: string
  vram_gb: number
  tier: string
  memory_hint: string
}

export interface DetailedHealth {
  status: string
  uptime_seconds: number
  timestamp: string
  system: {
    cpu_percent: number
    memory_percent: number
    memory_available_mb: number
  }
  gpu?: GPUInfo
  model_loaded: boolean
  model_type: string | null
  inference: {
    is_inferencing?: boolean
    inference_count?: number
    total_generated?: number
  }
}

export const systemController = {
  async getMetrics(): Promise<SystemMetrics> {
    return apiGet<SystemMetrics>('/system/metrics', undefined, { silent: true })
  },

  async getInfo(): Promise<SystemInfo> {
    return apiGet<SystemInfo>('/system/info', undefined, { silent: true })
  },

  async getDisk(): Promise<DiskUsage> {
    return apiGet<DiskUsage>('/system/disk', undefined, { silent: true })
  },

  async getDetailedHealth(): Promise<DetailedHealth> {
    return apiGet<DetailedHealth>('/health/detailed', undefined, { silent: true })
  },
}
