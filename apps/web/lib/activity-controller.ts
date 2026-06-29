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
    return apiPost(`${this.base}/data`, { body })
  }

  async train(body?: TrainRequest): Promise<TrainResponse> {
    return apiPost(`${this.base}/train`, { body: body || {} })
  }

  async predict(body: { data: number[][] }): Promise<PredictResponse> {
    return apiPost(`${this.base}/predict`, { body })
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
}

export const activityController = new ActivityController()
