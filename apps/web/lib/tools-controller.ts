import { apiGet } from './http-client'

export interface ToolParam {
  name: string
  type: string
  description: string
  required: boolean
}

export interface ToolInfo {
  name: string
  description: string
  parameters: ToolParam[]
  requires_approval: boolean
}

export interface ToolResult {
  tool: string
  status: 'success' | 'error' | 'executing'
  output: string
  error?: string
  duration_ms?: number
}

export const toolsController = {
  async list(): Promise<ToolInfo[]> {
    const data = await apiGet<{ tools: ToolInfo[] }>('/chat/tools')
    return data.tools || []
  },
}
