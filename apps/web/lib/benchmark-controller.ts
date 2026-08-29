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

export interface QualityMetrics {
  coherence_score: number
  quality_score: number
  repetition_rate: number
  total_responses: number
  avg_length: number
  empty_rate: number
}

export interface BenchmarkStatsData {
  total: number
  avg_tokens: number
  models: string[]
}

export const benchmarkController = {
  async metrics(model?: string): Promise<BenchmarkResult> {
    const q = model ? `?model=${encodeURIComponent(model)}` : ''
    return apiGet<BenchmarkResult>(`/benchmark/metrics${q}`)
  },

  async run(config: { model?: string; dataset?: string }): Promise<BenchmarkResult> {
    return apiPost<BenchmarkResult>('/benchmark/run', config)
  },

  async history(limit = 10): Promise<LoggedBenchmarkResponse[]> {
    const data = await apiGet<{ responses: LoggedBenchmarkResponse[]; count: number }>(`/benchmark/responses?limit=${limit}`)
    return data?.responses ?? []
  },

  async quality(): Promise<QualityMetrics> {
    const raw = await apiGet<{ responses_analyzed: number; metrics: { avg_length: number; length_std: number; repetition_rate: number; unique_bigram_ratio: number } }>('/benchmark/quality')
    const m = raw?.metrics ?? { avg_length: 0, length_std: 0, repetition_rate: 0, unique_bigram_ratio: 0 }
    const repRate = m.repetition_rate ?? 0
    const coherenceScore = m.unique_bigram_ratio ?? Math.max(0, 1 - repRate)
    const qualityScore = (coherenceScore + Math.max(0, 1 - repRate)) / 2
    return {
      coherence_score: coherenceScore,
      quality_score: qualityScore,
      repetition_rate: repRate,
      total_responses: raw?.responses_analyzed ?? 0,
      avg_length: m.avg_length ?? 0,
      empty_rate: 0,
    }
  },

  async stats(): Promise<BenchmarkStatsData> {
    const raw = await apiGet<{ total_responses: number; models: string[]; avg_length: number }>('/benchmark/stats')
    return {
      total: raw?.total_responses ?? 0,
      avg_tokens: raw ? Math.round(raw.avg_length / 5) : 0,
      models: raw?.models ?? [],
    }
  },
}
