import { apiPost } from '@/lib/http-client'

export interface ExtractFileResponse {
  text: string
  filename: string
  pages: number
  chars: number
  extension: string
}

export const filesController = {
  /** Upload a file (PDF, DOCX, TXT, etc.) and extract its text content. */
  async extract(file: File): Promise<ExtractFileResponse> {
    const fd = new FormData()
    fd.append('file', file)
    return apiPost<ExtractFileResponse>('/files/extract', fd, { raw: true })
  },
}
