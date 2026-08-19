/**
 * Files Controller — API for file management.
 *
 * Usage:
 *   import { filesController } from '@/lib/files-controller'
 *   const files = await filesController.list()
 *   await filesController.upload(formData)
 */

import { apiGet, apiPost, apiDelete } from './http-client'

export interface FileEntry {
  id: string
  filename: string
  size: number
  content_type: string
  uploaded_at: string
  ingested: boolean
  chunk_count?: number
}

export interface SearchResult {
  results: FileEntry[]
  count?: number
}

class FilesController {
  async list(): Promise<FileEntry[]> {
    const data = await apiGet<{ files?: FileEntry[] } | FileEntry[]>('/files/')
    if (!data) return []
    return Array.isArray(data) ? data : data.files ?? []
  }

  async upload(formData: FormData): Promise<{ filename?: string }> {
    return apiPost('/files/upload', formData, { raw: true })
  }

  async delete(id: string): Promise<void> {
    return apiDelete(`/files/${id}`)
  }

  async deleteBatch(ids: string[]): Promise<void> {
    await Promise.all(ids.map(id => apiDelete(`/files/${id}`)))
  }

  async ingest(id: string): Promise<void> {
    return apiPost(`/files/${id}/ingest`)
  }

  async search(query: string): Promise<FileEntry[]> {
    const data = await apiGet<{ results?: FileEntry[] } | FileEntry[]>(
      `/files/search?q=${encodeURIComponent(query)}`,
    )
    if (!data) return []
    return Array.isArray(data) ? data : data.results ?? []
  }
}

export const filesController = new FilesController()
