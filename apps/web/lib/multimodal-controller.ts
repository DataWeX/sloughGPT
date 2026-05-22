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

  async trainImage(dataUrl: string, fileName?: string): Promise<{ status: string; caption: string; confidence: number; images_learned: number }> {
    const blob = await dataUrlToBlob(dataUrl)
    const file = new File([blob], fileName || 'upload.png', { type: blob.type || 'image/png' })
    const fd = new FormData()
    fd.append('file', file)
    return apiPost('/multimodal/train', fd, { raw: true })
  },

  async trainBatch(files: File[]): Promise<{ status: string; job_id: string; total_images: number }> {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
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
}
