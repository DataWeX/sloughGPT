import { apiGet, apiPost, apiDelete } from '@/lib/http-client'

export interface FileItem {
  id: string
  filename: string
  extension: string
  size_bytes: number
  uploaded_at: number
  tags: string[]
}

export interface FileDetail {
  id: string
  filename: string
  extension: string
  size_bytes: number
  chars: number
  pages: number
  uploaded_at: number
  tags: string[]
  text: string
}

export interface UploadResponse {
  id: string
  filename: string
  chars: number
  pages: number
  size_bytes: number
}

export interface FileListResponse {
  files: FileItem[]
  total: number
}

export interface IngestResponse {
  id: string
  filename: string
  chars: number
  facts_stored: number
}

export const filesController = {
  async list(sort?: string, order?: string, tag?: string): Promise<FileListResponse> {
    const params = new URLSearchParams()
    if (sort) params.set('sort', sort)
    if (order) params.set('order', order)
    if (tag) params.set('tag', tag)
    const qs = params.toString()
    return apiGet<FileListResponse>(`/files${qs ? `?${qs}` : ''}`)
  },

  async extract(file: File): Promise<FileDetail> {
    const { id } = await this.upload(file)
    return this.get(id)
  },

  async upload(file: File, tags?: string[]): Promise<UploadResponse> {
    const fd = new FormData()
    fd.append('file', file)
    if (tags?.length) fd.append('tags', JSON.stringify(tags))
    return apiPost<UploadResponse>('/files/upload', fd, { raw: true })
  },

  async get(fileId: string): Promise<FileDetail> {
    return apiGet<FileDetail>(`/files/${fileId}`)
  },

  async delete(fileId: string): Promise<void> {
    return apiDelete(`/files/${fileId}`)
  },

  async search(q: string, tag?: string): Promise<FileListResponse> {
    const params = new URLSearchParams({ q })
    if (tag) params.set('tag', tag)
    return apiGet<FileListResponse>(`/files/search?${params}`)
  },

  async ingest(fileId: string): Promise<IngestResponse> {
    return apiPost<IngestResponse>(`/files/${fileId}/ingest`)
  },

  formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  },

  formatDate(ts: number): string {
    return new Date(ts * 1000).toLocaleString()
  },
}
