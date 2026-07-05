/** Map GET /info JSON to Monitoring UI state (real host + CUDA fields from API). */

export interface SystemInfo {
  platform: string
  python: string
  cpu_cores: number
  cpu_percent: number | null
  memory_total: number
  memory_used: number
  /** Host RAM; from used/total bytes when available so % matches the GB line. */
  memory_percent: number | null
  gpu_available: boolean
  gpu_name?: string
  gpu_memory?: number
  gpu_used?: number
  gpu_percent: number | null
  /** API process resident set size (bytes), when psutil reports it. */
  process_rss_bytes?: number | null
  /** True when API returned a ``host`` block (psutil-backed). */
  host_metrics_available: boolean
}

export interface InfoJson {
  pytorch_version?: string
  cuda_available?: boolean
  host?: {
    platform?: string
    platform_release?: string
    cpu_count_logical?: number
    cpu_percent?: number
    memory_total_bytes?: number
    memory_used_bytes?: number
    memory_percent?: number
    process_rss_bytes?: number | null
  }
  cuda?: {
    device?: string
    memory_total?: number
    memory_total_bytes?: number
    memory_used_bytes?: number
    memory_percent?: number
  }
}

/** Match the same bytes we show as GB — psutil `memory_percent` can disagree with `used`/`total`. */
function memoryPercentFromBytes(used: number, total: number): number | null {
  if (!(total > 0) || used < 0) return null
  return Math.min(100, Math.max(0, (used / total) * 100))
}

export function mapInfoToSystemInfo(data: InfoJson): SystemInfo {
  const pytorch = typeof data.pytorch_version === 'string' ? data.pytorch_version : 'N/A'
  const host = data.host
  const cuda = data.cuda

  let gpuMemory: number | undefined
  let gpuUsed: number | undefined
  let gpuPercent: number | null = null
  let gpuName: string | undefined

  if (cuda) {
    if (cuda.memory_total_bytes && cuda.memory_used_bytes != null) {
      gpuMemory = cuda.memory_total_bytes
      gpuUsed = cuda.memory_used_bytes
      gpuPercent = memoryPercentFromBytes(cuda.memory_used_bytes, cuda.memory_total_bytes)
    } else if (cuda.memory_total) {
      gpuMemory = cuda.memory_total * 1024 * 1024 * 1024
      gpuUsed = cuda.memory_total * 1024 * 1024 * 1024
      gpuPercent = cuda.memory_percent ?? null
    }
    gpuName = cuda.device
  }

  const cpuCores = host?.cpu_count_logical ?? 0
  const cpuPercent = host?.cpu_percent ?? null

  const hasHostMetrics = !!host
  const memoryTotal = host?.memory_total_bytes ?? 0
  const memoryUsed = host?.memory_used_bytes ?? 0
  const memoryPercent = hasHostMetrics && memoryTotal > 0
    ? memoryPercentFromBytes(memoryUsed, memoryTotal)
    : null

  const platform = host?.platform ?? 'Unavailable'
  const processRss = host?.process_rss_bytes ?? null
  const python = pytorch

  return {
    platform,
    python,
    cpu_cores: cpuCores,
    cpu_percent: cpuPercent,
    memory_total: memoryTotal,
    memory_used: memoryUsed,
    memory_percent: memoryPercent,
    gpu_available: !!data.cuda_available,
    gpu_name: gpuName,
    gpu_memory: gpuMemory,
    gpu_used: gpuUsed,
    gpu_percent: gpuPercent,
    process_rss_bytes: processRss,
    host_metrics_available: hasHostMetrics,
  }
}
