import { describe, expect, it } from 'vitest'

import { mapInfoToSystemInfo } from './monitoring-info'

describe('mapInfoToSystemInfo', () => {
  it('maps host block and CUDA bytes', () => {
    const s = mapInfoToSystemInfo({
      pytorch_version: '2.2.0',
      cuda_available: true,
      host: {
        platform: 'Darwin',
        platform_release: '24.0.0',
        cpu_count_logical: 8,
        cpu_percent: 12.5,
        memory_total_bytes: 16_000_000_000,
        memory_used_bytes: 8_000_000_000,
        memory_percent: 50,
        process_rss_bytes: 256_000_000,
      },
      cuda: {
        device: 'Test GPU',
        memory_total_bytes: 8_000_000_000,
        memory_used_bytes: 2_000_000_000,
        memory_percent: 25,
      },
    })
    expect(s.host_metrics_available).toBe(true)
    expect(s.cpu_percent).toBe(12.5)
    expect(s.memory_percent).toBe(50)
    expect(s.gpu_percent).toBe(25)
    expect(s.platform).toContain('Darwin')
    expect(s.process_rss_bytes).toBe(256_000_000)
  })

  it('derives host memory % from used/total bytes when psutil percent disagrees', () => {
    const s = mapInfoToSystemInfo({
      pytorch_version: '2.2.0',
      cuda_available: false,
      host: {
        platform: 'Darwin',
        cpu_count_logical: 8,
        cpu_percent: 25,
        memory_total_bytes: 16 * 1024 ** 3,
        memory_used_bytes: 8.2 * 1024 ** 3,
        memory_percent: 63,
      },
    })
    expect(s.memory_percent).toBeCloseTo((8.2 / 16) * 100, 5)
  })

  it('falls back when host is missing', () => {
    const s = mapInfoToSystemInfo({ pytorch_version: '2.0.0', cuda_available: false })
    expect(s.host_metrics_available).toBe(false)
    expect(s.cpu_percent).toBeNull()
  })
})
