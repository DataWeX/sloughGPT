import { apiGet, apiPost } from './http-client'

export interface SelfTrainStatus {
  status: string
  pid?: number
  returncode?: number
  history: string[]
}

export const selfTrainController = {
  async getStatus(): Promise<SelfTrainStatus> {
    const data = await apiGet<{ data?: SelfTrainStatus } | SelfTrainStatus>('/self-train/status')
    return (data as Record<string, unknown>).data != null ? (data as { data: SelfTrainStatus }).data : (data as SelfTrainStatus)
  },

  async start(opts?: { model?: string; temperature?: number; forever?: boolean }): Promise<{ status?: string; error?: string }> {
    const body: Record<string, unknown> = {}
    if (opts?.model) body.model = opts.model
    if (opts?.temperature != null) body.temperature = opts.temperature
    if (opts?.forever) body.forever = true
    const data = await apiPost('/self-train/start', body) as Record<string, unknown>
    const inner = data?.data != null ? data.data : data
    return inner as { status?: string; error?: string }
  },

  async stop(): Promise<void> {
    await apiPost('/self-train/stop', {})
  },
}
