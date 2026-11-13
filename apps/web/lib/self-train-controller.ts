import { apiGet, apiPost } from './http-client'

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

  async start(params?: { model?: string; temperature?: number; forever?: boolean }): Promise<{
    jobId: string
  }> {
    return apiPost<{ jobId: string }>('/self-train/start', params ?? {})
  },

  async stop(): Promise<{ status: string }> {
    return apiPost<{ status: string }>('/self-train/stop')
  },
}
