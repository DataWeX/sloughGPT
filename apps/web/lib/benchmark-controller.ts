import { apiGet, apiPost } from './http-client'

export interface BenchmarkResult {
  model: string
  model_loaded?: boolean
  model_id?: string
  inference_count?: number
  total_tokens?: number
  tokens_per_second?: number
  memory_mb: number
  num_parameters: number
  perplexity?: number
  bleu?: number
  latency_ms: number
  throughput: number
  throughput_tokens_per_sec: number
  inference_time_ms: number
  latency_p50_ms?: number
  latency_p95_ms?: number
  latency_p99_ms?: number
  error?: string
}

export interface LoggedBenchmarkResponse {
  timestamp: string
  user_message: string
  assistant_response: string
  model: string
  tokens_generated: number
  duration_ms: number
}

export const benchmarkController = {
  async run(config: { model?: string; dataset?: string }): Promise<BenchmarkResult> {
    return apiPost<BenchmarkResult>('/benchmark/run', config)
  },

  async history(limit = 10): Promise<LoggedBenchmarkResponse[]> {
    const data = await apiGet<{ responses: LoggedBenchmarkResponse[]; count: number }>(`/benchmark/responses?limit=${limit}`)
    return data?.responses ?? []
  },

  async quality(): Promise<{ coherence_score: number; quality_score: number; repetition_rate: number }> {
    return apiGet<{ coherence_score: number; quality_score: number; repetition_rate: number }>('/benchmark/quality')
  },

  async stats(): Promise<{ total: number; avg_tokens: number }> {
    return apiGet<{ total: number; avg_tokens: number }>('/benchmark/stats')
  },
}
