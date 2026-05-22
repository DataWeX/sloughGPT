import { apiClient } from './http-client'

export interface DownloadProgress {
  model_id: string
  status: 'queued' | 'downloading' | 'complete' | 'failed' | 'cancelled' | 'not_found' | 'already_cached' | 'already_downloading' | 'started'
  bytes_downloaded: number
  total_bytes: number
  speed_mb_per_sec: number
  eta_seconds: number
  percentage: number
  current_file: string
  error: string
  started_at: number
  completed_at: number
  files_completed: number
  files_total: number
  cached?: boolean
}

export async function startDownload(modelId: string, totalBytesHint = 0) {
  return apiClient.post('/models/download', { model_id: modelId, total_bytes_hint: totalBytesHint })
}

export async function getDownloadStatus(modelId: string): Promise<DownloadProgress> {
  const { data } = await apiClient.get(`/models/download/${encodeURIComponent(modelId)}`)
  return data as DownloadProgress
}

export async function listDownloads(): Promise<Record<string, DownloadProgress>> {
  const { data } = await apiClient.get('/models/downloads')
  return (data as { downloads: Record<string, DownloadProgress> }).downloads
}

export async function cancelDownload(modelId: string) {
  return apiClient.post(`/models/download/${encodeURIComponent(modelId)}/cancel`)
}
