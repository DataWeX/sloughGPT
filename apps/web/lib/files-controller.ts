import { apiPost } from '@/lib/http-client'

export interface ExtractFileResponse {
  text: string
  filename: string
  pages: number
  chars: number
  extension: string
}

export const filesController = {
  async extract(file: File): Promise<ExtractFileResponse> {
    const formData = new FormData()
    formData.append('file', file)
    return apiPost<ExtractFileResponse>('/files/extract', formData, { raw: true })
  },
}
