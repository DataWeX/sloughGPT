import { apiGet, apiPatch } from './http-client'

export interface GenerationConfig {
  temperature: number
  max_new_tokens: number
  top_p?: number
  top_k?: number
}

export const generationConfigController = {
  async get(): Promise<GenerationConfig> {
    return apiGet<GenerationConfig>('/config/generation')
  },

  async update(config: Partial<GenerationConfig>): Promise<void> {
    await apiPatch('/config/generation', config)
  },
}
