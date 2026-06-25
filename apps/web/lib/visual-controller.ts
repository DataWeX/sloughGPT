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

  async visualInference(imageBase64: string, prompt?: string): Promise<{ text: string; tokens_generated: number; elapsed_ms: number }> {
    return apiPost('/visual/generate', {
      image_base64: imageBase64,
      prompt: prompt || 'Describe this image in detail.',
      max_new_tokens: 256,
      temperature: 0.7,
      top_p: 0.9,
    })
  },

  async getVisualStatus(): Promise<{ loaded: boolean; model?: string; vision_encoder?: string; llm?: string }> {
    return apiGet('/visual/status')
  },

  async startVisualTrain(data: {
    data_path: string
    stage1_epochs?: number
    stage2_epochs?: number
    batch_size?: number
    learning_rate?: number
    lora_rank?: number
    output_dir?: string
  }): Promise<{ status: string; job_id: string; data_path: string; output_dir: string }> {
    return apiPost('/visual/train', data)
  },

  async getVisualTrainStatus(): Promise<{
    status: string
    job_id: string | null
    progress: number | null
    current_stage: string | null
    total_steps: number | null
    current_step: number | null
    current_loss: number | null
    result: Record<string, unknown> | null
    error: string | null
  }> {
    return apiGet('/visual/train/status')
  },

  async loadVisualModel(modelDir?: string): Promise<{ status: string; message: string }> {
    return apiPost('/visual/load', { model_dir: modelDir || 'models/visual-finetuned' })
  },

  async triggerDPO(data?: {
    max_pairs?: number
    learning_rate?: number
  }): Promise<{ status: string; message: string; job_id: string }> {
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

  // ── Visual Checkpoints ────────────────────────────────────────────

  async listCheckpoints(): Promise<{
    checkpoints: Array<{
      name: string; path: string; size_mb: number; created_at: string
      soul_name: string; lineage: string; llm: string
      final_loss: number | null; total_steps: number
      mean_accuracy: number | null; description: string
    }>
  }> {
    return apiGet('/visual/checkpoints')
  },

  async deleteCheckpoint(name: string): Promise<{ status: string; name: string }> {
    return apiDelete(`/visual/checkpoints/${encodeURIComponent(name)}`)
  },

  async loadCheckpoint(name: string): Promise<{ status: string; name: string; path: string }> {
    return apiPost(`/visual/checkpoints/${encodeURIComponent(name)}/load`)
  },

  async exportCheckpointXML(name: string): Promise<string> {
    const res = await fetch(`/visual/checkpoints/${encodeURIComponent(name)}/export-xml`)
    if (!res.ok) throw new Error('Export failed')
    return res.text()
  },

  // ── PDF Analysis ─────────────────────────────────────────────────

  async analyzePDF(data: {
    pdf_path: string
    question?: string
    per_page?: boolean
    max_new_tokens?: number
    temperature?: number
  }): Promise<{ status: string; analysis?: string; pages?: Array<{ page: number; text: string }> }> {
    return apiPost('/visual/pdf', {
      pdf_path: data.pdf_path,
      question: data.question || 'Summarize this document.',
      per_page: data.per_page ?? false,
      max_new_tokens: data.max_new_tokens ?? 512,
      temperature: data.temperature ?? 0.7,
    })
  },

  async analyzePDFUpload(file: File, question?: string, perPage?: boolean): Promise<{ status: string; analysis?: string; pages?: Array<{ page: number; text: string }> }> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('question', question || 'Summarize this document.')
    formData.append('per_page', String(perPage ?? false))
    formData.append('max_new_tokens', '512')
    formData.append('temperature', '0.7')

    const res = await fetch('/visual/pdf/upload', {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) {
      const err = await res.text().catch(() => res.statusText)
      throw new Error(`Upload failed (${res.status}): ${err}`)
    }
    return res.json()
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
}
