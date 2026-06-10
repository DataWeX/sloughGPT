/**
 * Dataset Controller — axios-based API for dataset management.
 */

import { apiGet, apiPost, apiPatch, apiDelete } from './http-client'
import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'

export type ImportSource = 'github' | 'huggingface' | 'url' | 'local' | 'kaggle' | 'csv' | 'isbn'

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
    return data.datasets || []
  },

  async get(id: string): Promise<Dataset | null> {
    try {
      return await apiGet<Dataset>(`/datasets/${id}`)
    } catch {
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
    const token = useAuthStore.getState().token
    const res = await fetch(`${PUBLIC_API_URL}/datasets/${id}/export`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
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

  async importFromGitHub(request: { url: string; name: string; extensions?: string[]; max_files?: number }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/github', request)
  },

  async importFromHuggingFace(request: { dataset_id: string; name?: string }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/huggingface', request)
  },

  async importFromURL(request: { url: string; name: string }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/url', request)
  },

  async importFromLocal(request: { path: string; name: string; extensions?: string[] }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/local', request)
  },

  async importFromKaggle(request: { dataset: string; name?: string }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/kaggle', request)
  },

  async importFromCSV(request: { url: string; name: string; delimiter?: string; encoding?: string }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/csv', request)
  },

  async importFromISBN(request: { isbn: string; name: string }): Promise<ImportResponse> {
    return apiPost<ImportResponse>('/datasets/import/isbn', request)
  },

  async batchImport(sources: string[]): Promise<{ imported: number; errors: string[] }> {
    return apiPost<{ imported: number; errors: string[] }>('/datasets/import/batch', { sources })
  },
}
