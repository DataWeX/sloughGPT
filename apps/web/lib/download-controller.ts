import { apiGet, apiPost } from './http-client'

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

export interface VerifyResult {
  status: 'verified' | 'corrupt' | 'not_cached' | 'error'
  model_id: string
  verified: boolean
  missing_files?: string[]
  missing_files_count?: number
  size_on_disk?: string
  error?: string
}

export async function startDownload(modelId: string, totalBytesHint = 0) {
  return apiPost('/models/download', { model_id: modelId, total_bytes_hint: totalBytesHint })
}

export async function getDownloadStatus(modelId: string): Promise<DownloadProgress> {
  return apiGet<DownloadProgress>(`/models/download/${encodeURIComponent(modelId)}`)
}

export async function listDownloads(): Promise<Record<string, DownloadProgress>> {
  const res = await apiGet<{ downloads: Record<string, DownloadProgress> }>('/models/downloads')
  return res.downloads
}

export async function cancelDownload(modelId: string) {
  return apiPost(`/models/download/${encodeURIComponent(modelId)}/cancel`)
}

export async function verifyDownload(modelId: string): Promise<VerifyResult> {
  return apiPost<VerifyResult>(`/models/download/${encodeURIComponent(modelId)}/verify`)
}

export async function retryDownload(modelId: string): Promise<{ status: string }> {
  return apiPost<{ status: string }>(`/models/download/${encodeURIComponent(modelId)}/retry`)
}
