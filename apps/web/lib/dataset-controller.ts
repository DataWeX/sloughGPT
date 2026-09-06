/**
 * Dataset Controller — axios-based API for dataset management.
 */

import { apiGet, apiPost, apiPatch, apiDelete, authFetch } from './http-client'
import { logger } from './dev-log'
import type { Method } from '@/hooks/useTrainingForm'

const _log = logger.child('dataset-controller')

export type ImportSource = 'github' | 'huggingface' | 'url' | 'local' | 'kaggle' | 'csv' | 'isbn'
export type DatasetFormat = 'jsonl' | 'csv' | 'json' | 'messages' | 'dialogue' | 'text'

export interface DatasetStats {
  format: DatasetFormat
  samples: number
  chars: number
  avg_length: number
  has_messages: boolean
  sample_preview: string[]
  lines: number
  suggested_method: Method | 'unknown'
  file_type: string
  error?: string
}

export interface TrainingConfig {
  method: Method
  model: string
  epochs: number
  learning_rate: number
  batch_size: number
  use_lora: boolean
  lora_rank: number
  max_seq_length: number
  reasoning: string
}

export interface DatasetPreview {
  dataset_id: string
  samples: Array<{ path: string; language: string; content: string; size: number }>
  total_samples: number
  total_chars: number
  languages: Record<string, number>
}

export interface ImportResponse {
  success: boolean
  dataset_id: string
  message: string
  output_path: string
}

export interface Dataset {
  id: string
  name: string
  source: string
  type?: string
  size: number
  samples?: number
  created_at: string
  updated_at?: string
  tags?: string[]
  vlm_metadata?: {
    type: string
    image_dir: string
    image_count: number
    auto_captioned: boolean
  }
}

export interface GitHubRepo {
  id: string
  name: string
  full_name: string
  description: string | null
  stars: number
  url: string
  language: string | null
}

export interface BookResult {
  key: string
  title: string
  author: string
  isbn: string
  year: number | null
  cover: number | null
}

export const datasetController = {
  async list(): Promise<Dataset[]> {
    const data = await apiGet<{ datasets: Dataset[] }>('/datasets')
    return data?.datasets ?? []
  },

  async search(query: string): Promise<Dataset[]> {
    const data = await apiGet<{ results: Dataset[] }>(`/datasets/search?q=${encodeURIComponent(query)}`)
    return data?.results ?? []
  },

  async get(id: string): Promise<Dataset | null> {
    try {
      return await apiGet<Dataset>(`/datasets/${id}`)
    } catch (err) {
      _log.debug('Failed to get dataset', { error: err instanceof Error ? err.message : String(err) })
      return null
    }
  },

  async create(config: { name: string; description?: string }): Promise<Dataset> {
    return apiPost<Dataset>('/datasets', config)
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/datasets/${id}`)
  },

  async addData(id: string, data: string[]): Promise<void> {
    await apiPost(`/datasets/${id}/data`, { data })
  },

  async update(id: string, updates: Partial<Dataset>): Promise<Dataset> {
    return apiPatch<Dataset>(`/datasets/${id}`, updates)
  },

  async export(id: string, format: string = 'jsonl'): Promise<Blob> {
    const res = await authFetch(`/datasets/${id}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format }),
    })
    return res.blob()
  },

  async preview(id: string, limit = 10): Promise<DatasetPreview> {
    return apiGet<DatasetPreview>(`/datasets/${id}/preview?limit=${limit}`)
  },

  async searchGitHubRepos(query: string, limit = 10): Promise<{ repos: GitHubRepo[] }> {
    return apiGet<{ repos: GitHubRepo[] }>(`/datasets/search/github`, { q: query, limit: String(limit) })
  },

  async searchBooks(query: string, limit = 10): Promise<{ books: BookResult[] }> {
    return apiGet<{ books: BookResult[] }>(`/datasets/search/books`, { q: query, limit: String(limit) })
  },

  async importFromGitHub(request: { url: string; name: string; extensions?: string[]; max_files?: number }, opts?: { signal?: AbortSignal }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/github', request, { signal: opts?.signal })
  },

  async importFromHuggingFace(request: { dataset_id: string; name?: string }, opts?: { signal?: AbortSignal }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/huggingface', request, { signal: opts?.signal })
  },

  async importFromURL(request: { url: string; name: string }, opts?: { signal?: AbortSignal }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/url', request, { signal: opts?.signal })
  },

  async importFromLocal(request: { path: string; name: string; extensions?: string[] }, opts?: { signal?: AbortSignal }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/local', request, { signal: opts?.signal })
  },

  async importFromKaggle(request: { dataset: string; name?: string }, opts?: { signal?: AbortSignal }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/kaggle', request, { signal: opts?.signal })
  },

  async importFromCSV(request: { url: string; name: string; delimiter?: string; encoding?: string }, opts?: { signal?: AbortSignal }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/csv', request, { signal: opts?.signal })
  },

  async importFromISBN(request: { isbn: string; name: string }, opts?: { signal?: AbortSignal }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/isbn', request, { signal: opts?.signal })
  },

  async batchImport(sources: string[]): Promise<{ imported: number; errors: string[] }> {
    return apiPost<{ imported: number; errors: string[] }>('/datasets/import/batch', { sources })
  },

  async convertToMessages(datasetId: string, systemPrompt: string = "You are a helpful assistant."): Promise<{
    status: string
    new_dataset_id: string
    total_conversations: number
  }> {
    return apiPost(`/datasets/convert-to-messages?dataset_id=${encodeURIComponent(datasetId)}&system_prompt=${encodeURIComponent(systemPrompt)}`)
  },

  async getStats(datasetId: string): Promise<DatasetStats> {
    return apiGet<DatasetStats>(`/datasets/${encodeURIComponent(datasetId)}/stats`)
  },

  async createVersion(datasetId: string): Promise<{ timestamp: string; message: string }> {
    return apiPost(`/datasets/${encodeURIComponent(datasetId)}/versions`)
  },

  async listVersions(datasetId: string): Promise<{ versions: string[]; count: number }> {
    return apiGet(`/datasets/${encodeURIComponent(datasetId)}/versions`)
  },

  async restoreVersion(datasetId: string, timestamp: string): Promise<{ success: boolean; message: string }> {
    return apiPost(`/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(timestamp)}`)
  },

  async createFromChat(params: {
    messages: Array<{ role: string; content: string }>
    name?: string
  }): Promise<{ status: string; dataset_id: string; name: string; messages_exported: number }> {
    return apiPost('/datasets/from-chat', params)
  },
}
