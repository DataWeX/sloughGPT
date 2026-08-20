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
    const d = await apiGet<LoraEvalHistory>(`/lora-eval/history?limit=${limit}`)
    return d?.results ?? []
  },
}
