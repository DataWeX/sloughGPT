import { apiGet, apiPost, apiDelete } from './http-client'

export interface SensorWindow {
  data: number[][]
  label?: number
}

export interface TrainRequest {
  epochs: number
  lr: number
  batch_size: number
}

export interface TrainResponse {
  status: string
  epochs: number
  final_loss: number | null
  val_accuracy: number | null
  num_samples: number
  message: string
}

export interface PredictResponse {
  activity: string
  class_id: number
  confidence: number
  probabilities: number[]
}

export interface ActivityStatus {
  model_loaded: boolean
  num_recordings: number
  num_labels: number
  activities: string[]
  device: string
}

export interface DatasetRecord {
  id: number
  path: string
  samples: number
  label: number
  activity: string
}

export interface DatasetResponse {
  recordings: DatasetRecord[]
  total: number
}

export interface DataResponse {
  id: number
  path: string
  samples: number
}

class ActivityController {
  private base = '/activity'

  async recordData(body: SensorWindow): Promise<DataResponse> {
    return apiPost(`${this.base}/data`, body)
  }

  async train(body?: TrainRequest): Promise<TrainResponse> {
    return apiPost(`${this.base}/train`, body || {})
  }

  async predict(body: { data: number[][] }): Promise<PredictResponse> {
    return apiPost(`${this.base}/predict`, body)
  }

  async status(): Promise<ActivityStatus> {
    return apiGet(this.base + '/status')
  }

  async dataset(): Promise<DatasetResponse> {
    return apiGet(this.base + '/dataset')
  }

  async deleteAll(): Promise<{ deleted: number }> {
    return apiDelete(this.base + '/data')
  }

  /**
   * Stream training progress via SSE. Yields per-epoch events with loss/accuracy
   * and a final "complete" event. Throws on connection error.
   */
  async *trainStream(body?: TrainRequest): AsyncGenerator<Record<string, unknown>, void, unknown> {
    const url = `${await import('./config').then(m => m.PUBLIC_API_URL)}${this.base}/train/stream`
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    })
    if (!res.ok) throw new Error(`Train SSE returned ${res.status}`)
    const reader = res.body?.getReader()
    if (!reader) throw new Error('No response body')
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (trimmed.startsWith('data: ')) {
          yield JSON.parse(trimmed.slice(6))
        }
      }
    }
  }

  /** Download trained model.npz as a Blob. */
  async downloadModel(): Promise<Blob> {
    const url = `${this.base}/model`
    const res = await fetch(url)
    if (!res.ok) throw new Error(`Model download returned ${res.status}`)
    return res.blob()
  }
}

export const activityController = new ActivityController()
export type { ActivityController }
