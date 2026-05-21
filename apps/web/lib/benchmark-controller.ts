import { apiGet, apiPost } from './http-client'

export interface BenchmarkResult {
  model: string
  perplexity?: number
  bleu?: number
  latency_ms: number
  throughput: number
  num_parameters: number
  memory_mb: number
  throughput_tokens_per_sec: number
  inference_time_ms: number
  latency_p50_ms?: number
  latency_p95_ms?: number
  latency_p99_ms?: number
  error?: string
}

export const benchmarkController = {
  async run(config: { model?: string; dataset?: string }): Promise<BenchmarkResult> {
    return apiPost<BenchmarkResult>('/benchmark/run', config)
  },

  async history(limit = 10): Promise<BenchmarkResult[]> {
    const data = await apiGet<{ results: BenchmarkResult[] } | BenchmarkResult[]>(`/benchmark/metrics?limit=${limit}`)
    return Array.isArray(data) ? data : data.results || []
  },

  async quality(): Promise<{ coherence_score: number; quality_score: number; repetition_rate: number }> {
    return apiGet<{ coherence_score: number; quality_score: number; repetition_rate: number }>('/benchmark/quality')
  },

  async stats(): Promise<{ total: number; avg_tokens: number }> {
    return apiGet<{ total: number; avg_tokens: number }>('/benchmark/stats')
  },
}
