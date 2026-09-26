/**
 * Files Controller — API for file management.
 *
 * Usage:
 *   import { filesController } from '@/lib/files-controller'
 *   const files = await filesController.list()
 *   await filesController.upload(formData)
 */

import { apiGet, apiPost, apiDelete } from './http-client'
import { toDateSeconds } from './time-format'

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

export interface FileDetail extends FileEntry {
  chars?: number
  pages?: number
  text?: string
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
  chars?: number
  uploaded_at: string | number
  tags?: string[]
}

/**
 * Backend stores `uploaded_at` as epoch seconds (float), but renderers parse
 * ISO strings (`new Date("1767225600")` is Invalid Date). Convert numeric
 * values here; ISO/date-only strings pass through; missing/zero values become
 * `''` so the page renders its '—' fallback instead of garbage.
 */
function normalizeUploadedAt(value: string | number | null | undefined): string {
  if (value == null) return ''
  if (typeof value === 'number') {
    return value > 0 ? (toDateSeconds(value)?.toISOString() ?? '') : ''
  }
  const trimmed = value.trim()
  if (trimmed === '') return ''
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    const n = Number(trimmed)
    return n > 0 ? (toDateSeconds(n)?.toISOString() ?? '') : ''
  }
  return value
}

function mapFileEntry(f: BackendFileItem): FileEntry {
  return {
    id: f.id,
    filename: f.filename,
    size: f.size_bytes ?? 0,
    content_type: f.extension ?? 'unknown',
    uploaded_at: normalizeUploadedAt(f.uploaded_at),
    ingested: (f.chars ?? 0) > 0 || (f.tags?.length ?? 0) > 0,
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
    await Promise.all(ids.map((id) => apiDelete(`/files/${id}`)))
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

  async getDetail(id: string): Promise<FileDetail | null> {
    try {
      const data = await apiGet<Record<string, unknown>>(`/files/${id}`)
      if (!data) return null
      return {
        ...mapFileEntry(data as unknown as BackendFileItem),
        chars: data.chars as number | undefined,
        pages: data.pages as number | undefined,
        text: data.text as string | undefined,
      }
    } catch {
      return null
    }
  }
}

export const filesController = new FilesController()
