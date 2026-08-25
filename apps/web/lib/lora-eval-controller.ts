/**
 * LoRA Eval controller — adapter quality evaluation.
 */
import { apiGet, apiPost } from '@/lib/http-client'

export interface LoraEvalResult {
  status: string
  adapter_path?: string
  verdict?: string
  timestamp?: string
  perplexity?: number
  bleu?: number
  baseline?: Record<string, unknown>
  with_adapter?: Record<string, unknown>
  delta?: Record<string, unknown>
  report?: string
  elapsed_ms?: number
}

export type LoraEvalHistory = LoraEvalResult[]

export interface LoraAggregationResult {
  status: string
  output_path?: string
  user_count?: number
  total_feedback?: number
  eval?: {
    verdict: string
    perplexity_delta?: number
    bleu_delta?: number
    throughput_delta?: number
    report?: string
  }
}

export const loraEvalController = {
  async runEval(adapterPath: string, soul?: string): Promise<LoraEvalResult> {
    const params = new URLSearchParams({ adapter_path: adapterPath })
    if (soul) params.set('soul', soul)
    return apiGet<LoraEvalResult>(`/lora-eval/run?${params}`)
  },

  async getHistory(limit = 10): Promise<LoraEvalResult[]> {
    const data = await apiGet<{ results?: LoraEvalResult[] } | LoraEvalResult[]>(`/lora-eval/history?limit=${limit}`)
    if (Array.isArray(data)) return data
    return data.results ?? []
  },

  async aggregate(topK = 10, minFeedback = 5, outputName = 'best_aggregated'): Promise<LoraAggregationResult> {
    const params = new URLSearchParams({
      top_k: String(topK),
      min_feedback: String(minFeedback),
      output_name: outputName,
      run_eval: 'true',
    })
    return apiPost(`/lora-eval/aggregate?${params}`)
  },
}
