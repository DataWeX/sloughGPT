import { apiGet, apiPost, apiDelete } from './http-client'

export const visualController = {
  async createVisualDataset(name: string, imageDir: string, captionPrompt?: string, autoCaption = true): Promise<{ status: string; dataset: string; path: string; entries: number; auto_captioned: boolean }> {
    return apiPost('/multimodal/visual-dataset', {
      name,
      image_dir: imageDir,
      caption_prompt: captionPrompt || 'Describe this image in detail.',
      auto_caption: autoCaption,
    })
  },

  async getVisualStatus(): Promise<{ video_training: Record<string, unknown>; dpo: Record<string, unknown> }> {
    return apiGet('/visual/status')
  },

  async triggerDPO(data?: {
    max_pairs?: number
    learning_rate?: number
  }): Promise<{ status: string; steps: number; avg_loss: number | null; ppl_before: number | null; ppl_after: number | null; ppl_delta_pct: number | null; pairs_trained: number; elapsed_seconds: number }> {
    return apiPost('/visual/dpo', data || {})
  },

  async getDPOStatus(): Promise<{
    status: string
    last_run: string | null
    result: Record<string, unknown> | null
    accepted_count: number
    rejected_count: number
  }> {
    return apiGet('/visual/dpo/status')
  },

  // ── Video Training ──────────────────────────────────────────────

  async startVideoTrain(data: {
    data_path: string
    epochs?: number
    batch_size?: number
    learning_rate?: number
    output_dir?: string
  }): Promise<{ status: string; job_id: string; data_path: string; output_dir: string }> {
    return apiPost('/visual/train-video', data)
  },

  async getVideoTrainStatus(): Promise<{
    status: string
    job_id: string | null
    current_epoch: number
    current_step: number
    total_steps: number
    current_loss: number | null
    result: Record<string, unknown> | null
    error: string | null
  }> {
    return apiGet('/visual/train-video/status')
  },

  async videoInference(data: {
    video_path: string
    max_len?: number
    temperature?: number
  }): Promise<{ text: string; checkpoint: string; elapsed_ms: number }> {
    return apiPost('/visual/video-infer', data)
  },

  // ── Backward-compat aliases (legacy pages) ──────────────────────

  async getVisualTrainStatus(): Promise<{ status: string; job_id: string | null; current_epoch: number; current_step: number; total_steps: number; current_loss: number | null; result: Record<string, unknown> | null; error: string | null }> {
    return apiGet('/visual/train-video/status')
  },

  async startVisualTrain(data: { data_path: string; epochs?: number; batch_size?: number; learning_rate?: number; output_dir?: string }): Promise<{ status: string; job_id: string; data_path: string; output_dir: string }> {
    return apiPost('/visual/train-video', data)
  },

  async loadVisualModel(modelId: string): Promise<{ status: string; message: string }> {
    return apiPost('/visual/load', { model_id: modelId })
  },

  async visualInference(data: { image_path: string; max_len?: number; temperature?: number }): Promise<{ text: string; checkpoint?: string; elapsed_ms: number }> {
    return apiPost('/visual/video-infer', { video_path: data.image_path, max_len: data.max_len, temperature: data.temperature })
  },

  async listCheckpoints(): Promise<any[]> {
    return apiGet('/visual/checkpoints')
  },

  async loadCheckpoint(name: string): Promise<Record<string, unknown>> { return apiPost(`/visual/checkpoints/${encodeURIComponent(name)}/load`) },

  async deleteCheckpoint(name: string): Promise<Record<string, unknown>> { return apiDelete(`/visual/checkpoints/${encodeURIComponent(name)}`) },

  async analyzePDFUpload(_file: File, _question?: string, _perPage?: boolean): Promise<{ analysis?: string; pages?: Array<{ text: string }> }> {
    return { analysis: '', pages: [] }
  },

  async analyzePDF(_params: { pdf_path: string; question?: string; per_page?: boolean }): Promise<{ analysis?: string; pages?: Array<{ text: string }> }> {
    return { analysis: '', pages: [] }
  },
}
