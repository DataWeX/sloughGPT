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
  extension?: string
  tags?: string[]
}

export interface SearchResult {
  results: FileEntry[]
  count?: number
}

interface BackendFileItem {
  id: string
  filename: string
  extension?: string
  size_bytes?: number
  uploaded_at: string | number
  tags?: string[]
}

function mapFileEntry(f: BackendFileItem): FileEntry {
  return {
    id: f.id,
    filename: f.filename,
    size: f.size_bytes ?? 0,
    content_type: f.extension ?? 'unknown',
    uploaded_at: String(f.uploaded_at),
    ingested: (f.tags?.length ?? 0) > 0,
    extension: f.extension,
    tags: f.tags,
  }
}

class FilesController {
  async list(): Promise<FileEntry[]> {
    const data = await apiGet<{ files?: BackendFileItem[] } | BackendFileItem[]>('/files/')
    if (!data) return []
    const raw = Array.isArray(data) ? data : (data.files ?? [])
    return raw.map(mapFileEntry)
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
    const data = await apiGet<{ files?: BackendFileItem[] }>(
      `/files/search?q=${encodeURIComponent(query)}`,
    )
    if (!data) return []
    return (data.files ?? []).map(mapFileEntry)
  }
}

export const filesController = new FilesController()
