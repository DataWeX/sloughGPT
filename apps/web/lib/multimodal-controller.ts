import { apiGet, apiPost } from './http-client'

export interface MultimodalCapabilities {
  speech_to_text: boolean
  image_caption: boolean
  speech_model: string | null
  vision_model: string | null
  images_learned: number
  trained: boolean
  replay_buffer_size: number
  learning_method: string
  background_job_running: boolean
  status: string
}

export interface LearningProgress {
  images_learned: number
  trained: boolean
  vocab_size: number
  replay_buffer_size: number
}

export interface TrainingReport {
  images_learned: number
  vocab_size: number
  replay_buffer_size: number
  caption_history: string[]
  unique_captions: number
  diversity_ratio: number
  trained: boolean
  accuracy_history: number[]
  mean_accuracy: number
  last_accuracy: number
}

export interface TrainingStatus {
  running: boolean
  job_id: string | null
  total: number
  completed: number
  errors: number
  progress_pct: number
  current_caption: string
  current_image: string
  started_at: string | null
  finished_at: string | null
}

async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl)
  return res.blob()
}

export const multimodalController = {
  async getCapabilities(): Promise<MultimodalCapabilities> {
    return apiGet<MultimodalCapabilities>('/multimodal/capabilities')
  },

  async getLearningProgress(): Promise<LearningProgress> {
    return apiGet<LearningProgress>('/multimodal/learning-progress')
  },

  async getTrainingReport(): Promise<TrainingReport> {
    return apiGet<TrainingReport>('/multimodal/training-report')
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    return apiGet<TrainingStatus>('/multimodal/training-status')
  },

  async trainImage(dataUrl: string, fileName?: string, label?: string): Promise<{ status: string; caption: string; confidence: number; images_learned: number; accuracy: number; supervised: boolean }> {
    const blob = await dataUrlToBlob(dataUrl)
    const file = new File([blob], fileName || 'upload.png', { type: blob.type || 'image/png' })
    const fd = new FormData()
    fd.append('file', file)
    if (label) fd.append('label', label)
    return apiPost('/multimodal/train', fd, { raw: true })
  },

  async trainBatch(files: File[]): Promise<{ status: string; job_id: string; total_images: number }> {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    return apiPost('/multimodal/train-batch', fd, { raw: true })
  },

  async trainBatchFromDir(datasetPath: string): Promise<{ status: string; job_id: string; total_images: number }> {
    const fd = new FormData()
    fd.append('dataset_path', datasetPath)
    return apiPost('/multimodal/train-batch', fd, { raw: true })
  },

  async transcribeAudio(file: File): Promise<{ text: string; error?: string }> {
    const fd = new FormData()
    fd.append('file', file)
    return apiPost('/multimodal/transcribe', fd, { raw: true })
  },

  async generateImage(prompt: string, steps: number = 20, guidanceScale: number = 7.5): Promise<{ status: string; image: string; prompt: string }> {
    const fd = new FormData()
    fd.append('prompt', prompt)
    fd.append('steps', String(steps))
    fd.append('guidance_scale', String(guidanceScale))
    return apiPost('/multimodal/generate-image', fd, { raw: true })
  },

  async getGenerationStatus(): Promise<{ models_loaded: boolean; capabilities: Record<string, boolean | string> }> {
    return apiGet('/multimodal/generation-status')
  },

  async processVideo(file: File, numFrames: number = 16): Promise<{ status: string; caption: string; num_frames: number }> {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('num_frames', String(numFrames))
    return apiPost('/multimodal/process-video', fd, { raw: true })
  },

  async synthesizeSpeech(text: string): Promise<{ status: string; audio: string; text: string; duration_sec: number }> {
    const fd = new FormData()
    fd.append('text', text)
    return apiPost('/multimodal/synthesize-speech', fd, { raw: true })
  },

  async createVLMDataset(name: string, imageDir: string, captionPrompt?: string, autoCaption = true): Promise<{ status: string; dataset: string; path: string; entries: number; auto_captioned: boolean }> {
    return apiPost('/multimodal/vlm-dataset', {
      name,
      image_dir: imageDir,
      caption_prompt: captionPrompt || 'Describe this image in detail.',
      auto_caption: autoCaption,
    })
  },

  async vlmInference(imageBase64: string, prompt?: string): Promise<{ text: string; tokens_generated: number; elapsed_ms: number }> {
    return apiPost('/vlm/generate', {
      image_base64: imageBase64,
      prompt: prompt || 'Describe this image in detail.',
      max_new_tokens: 256,
      temperature: 0.7,
      top_p: 0.9,
    })
  },

  async getVLMStatus(): Promise<{ loaded: boolean; model?: string; vision_encoder?: string; llm?: string }> {
    return apiGet('/vlm/status')
  },

  async startVLMTrain(data: {
    data_path: string
    stage1_epochs?: number
    stage2_epochs?: number
    batch_size?: number
    learning_rate?: number
    lora_rank?: number
    output_dir?: string
  }): Promise<{ status: string; job_id: string; data_path: string; output_dir: string }> {
    return apiPost('/vlm/train', data)
  },

  async getVLMTrainStatus(): Promise<{
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
    return apiGet('/vlm/train/status')
  },

  async loadVLMModel(modelDir?: string): Promise<{ status: string; message: string }> {
    return apiPost('/vlm/load', { model_dir: modelDir || 'models/vlm-finetuned' })
  },

  async triggerDPO(data?: {
    max_pairs?: number
    learning_rate?: number
  }): Promise<{ status: string; message: string; job_id: string }> {
    return apiPost('/vlm/dpo', data || {})
  },

  async getDPOStatus(): Promise<{
    status: string
    last_run: string | null
    result: Record<string, unknown> | null
    accepted_count: number
    rejected_count: number
  }> {
    return apiGet('/vlm/dpo/status')
  },

  async analyzeImage(file: File): Promise<{
    caption: string
    confidence: number
    tags: string[]
    accuracy: number
    supervised: boolean
    images_learned: number
    trained: boolean
    replay_buffer_size: number
    mean_accuracy: number
  }> {
    const fd = new FormData()
    fd.append('file', file)
    return apiPost('/multimodal/analyze', fd, { raw: true })
  },

  async resetModel(): Promise<{ status: string; message: string }> {
    return apiPost('/multimodal/reset')
  },
}
