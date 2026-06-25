import { apiGet, apiPost } from './http-client'

export interface ExportRequest {
  output_path?: string
  format?: string
  include_tokenizer?: boolean
}

export interface ExportResult {
  status: string
  format: string
  files?: string[]
  error?: string
}

export const exportController = {
  async exportModel(data: ExportRequest): Promise<ExportResult> {
    return apiPost<ExportResult>('/models/export', {
      output_path: 'models/exported',
      format: 'sou',
      include_tokenizer: true,
      ...data,
    })
  },

  async getFormats(): Promise<string[]> {
    const res = await apiGet<{ formats: string[] }>('/models/export/formats')
    return res.formats
  },
}
