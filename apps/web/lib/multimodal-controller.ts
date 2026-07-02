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
}

interface UnifiedStatus {
  engine: {
    speech_to_text: boolean
    image_caption: boolean
    speech_model: string | null
    vision_model: string | null
    status: string
  }
  learning: {
    images_learned: number
    trained: boolean
    vocab_size: number
    replay_buffer_size: number
    learning_method: string
    caption_history: string[]
    unique_captions: number
    diversity_ratio: number
    accuracy_history: number[]
    mean_accuracy: number
    last_accuracy: number
  }
  batch: TrainingStatus
  dpo: { status: string; last_run: string | null; result: any; accepted_count: number; rejected_count: number }
  video: { status: string; job_id: string | null; current_epoch: number; current_step: number; total_steps: number; current_loss: number | null; result: any; error: string | null }
}

async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl)
  return res.blob()
}

export const multimodalController = {
  async getStatus(): Promise<UnifiedStatus> {
    return apiGet<UnifiedStatus>('/multimodal/status')
  },

  async getCapabilities(): Promise<MultimodalCapabilities> {
    const s = await this.getStatus() as UnifiedStatus
    return {
      speech_to_text: s.engine.speech_to_text,
      image_caption: s.engine.image_caption,
      speech_model: s.engine.speech_model,
      vision_model: s.engine.vision_model,
      images_learned: s.learning.images_learned,
      trained: s.learning.trained,
      replay_buffer_size: s.learning.replay_buffer_size,
      learning_method: s.learning.learning_method,
      background_job_running: s.batch.running,
      status: s.engine.status,
    }
  },

  async getLearningProgress(): Promise<LearningProgress> {
    const s = await this.getStatus() as UnifiedStatus
    return {
      images_learned: s.learning.images_learned,
      trained: s.learning.trained,
      vocab_size: s.learning.vocab_size,
      replay_buffer_size: s.learning.replay_buffer_size,
    }
  },

  async getTrainingReport(): Promise<TrainingReport> {
    const s = await this.getStatus() as UnifiedStatus
    return {
      images_learned: s.learning.images_learned,
      vocab_size: s.learning.vocab_size,
      replay_buffer_size: s.learning.replay_buffer_size,
      caption_history: s.learning.caption_history,
      unique_captions: s.learning.unique_captions,
      diversity_ratio: s.learning.diversity_ratio,
      trained: s.learning.trained,
      accuracy_history: s.learning.accuracy_history,
      mean_accuracy: s.learning.mean_accuracy,
      last_accuracy: s.learning.last_accuracy,
    }
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    const s = await this.getStatus() as UnifiedStatus
    return s.batch
  },

  async getDPOStatus(): Promise<{ status: string; accepted_count: number; rejected_count: number; result?: any }> {
    const s = await this.getStatus() as UnifiedStatus
    return s.dpo
  },

  async getVideoStatus(): Promise<{ status: string; job_id: string | null; current_epoch: number; current_step: number; total_steps: number; current_loss: number | null }> {
    const s = await this.getStatus() as UnifiedStatus
    return s.video
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

  async analyzeImage(file: File, prompt?: string): Promise<{
    caption: string; confidence: number; tags: string[]; accuracy: number;
    supervised: boolean; images_learned: number; trained: boolean;
    replay_buffer_size: number; mean_accuracy: number;
  }> {
    const fd = new FormData()
    fd.append('file', file)
    if (prompt) fd.append('prompt', prompt)
    return apiPost('/multimodal/analyze', fd, { raw: true })
  },

  async resetModel(): Promise<{ status: string; message: string }> {
    return apiPost('/multimodal/reset')
  },
}
