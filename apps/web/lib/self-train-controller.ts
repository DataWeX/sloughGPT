import { apiGet } from './http-client'

export interface SelfTrainStatus {
  jobId: string
  phase: 'idle' | 'generating' | 'distilling' | 'training' | 'evaluating' | 'deploying' | 'complete'
  progress: number
  currentStep?: string
  totalSteps?: number
  startedAt?: string
  completedAt?: string
}

export const selfTrainController = {
  async getStatus(): Promise<SelfTrainStatus> {
    return apiGet<SelfTrainStatus>('/self-train/status')
  },

  async start(): Promise<{ jobId: string }> {
    return apiGet<{ jobId: string }>('/self-train/start', {}, { method: 'POST' })
  },
}
