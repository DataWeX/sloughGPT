'use client'

import { apiGet, apiPost } from './http-client'

export interface MetaWeights {
  temperature: number
  repetition_penalty: number
  top_p: number
  top_k: number
  style_bias: number
  confidence_boost: number
  based_on_samples: number
}

export interface MetaWeightStats {
  history_length: number
  avg_temperature?: number
  avg_top_p?: number
  avg_repetition_penalty?: number
  quality_trend?: { step: number; quality: number }[]
}

export const metaWeightsController = {
  async getWeights(userMessage: string, k = 5, userId = 'default'): Promise<MetaWeights> {
    return apiPost('/meta-weights/get', { user_message: userMessage, k, user_id: userId })
  },

  async getStats(): Promise<MetaWeightStats> {
    return apiGet('/meta-weights/stats')
  },

  async ping(): Promise<{ status: string }> {
    return apiGet('/meta-weights/ping')
  },
}
