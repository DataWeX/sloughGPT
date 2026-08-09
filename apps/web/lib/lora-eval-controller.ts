import { apiGet } from './http-client'

export interface LoraEvalResult {
  adapter_path: string
  verdict: string
  perplexity?: number
  bleu?: number
  timestamp?: string
}

export interface LoraEvalHistory {
  results: LoraEvalResult[]
}

export const loraEvalController = {
  async runEval(adapterPath: string): Promise<unknown> {
    return apiGet(`/lora-eval/run?adapter_path=${encodeURIComponent(adapterPath)}`)
  },

  async getHistory(limit = 10): Promise<LoraEvalResult[]> {
    const data = await apiGet<{ data?: LoraEvalHistory } | LoraEvalHistory>(`/lora-eval/history?limit=${limit}`)
    const d = (data as Record<string, unknown>).data != null ? (data as { data: LoraEvalHistory }).data : (data as LoraEvalHistory)
    return d?.results ?? []
  },
}
