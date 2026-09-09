import { apiGet, apiPost } from './http-client'

export interface CloudJob {
  job_id: string
  provider: string
  status: string
  progress: number
  error?: string
}

export interface PluginInfo {
  name: string
  version: string
  description: string
  author: string
  enabled: boolean
  hooks: string[]
}

export const pluginsCloudController = {
  async listCloudJobs(limit: number = 10): Promise<CloudJob[]> {
    const resp = await apiGet<{ jobs: CloudJob[] }>(`/cloud-training/jobs?limit=${limit}`)
    return resp.jobs
  },

  async submitCloudJob(provider: string, datasetId: string): Promise<{ job_id: string; status: string }> {
    return apiPost('/cloud-training/submit', { provider, dataset_id: datasetId })
  },

  async cloudJobStatus(jobId: string): Promise<CloudJob> {
    return apiGet(`/cloud-training/${jobId}/status`)
  },

  async cancelCloudJob(jobId: string): Promise<{ cancelled: boolean }> {
    return apiPost(`/cloud-training/${jobId}/cancel`)
  },

  async listPlugins(): Promise<PluginInfo[]> {
    const resp = await apiGet<{ plugins: PluginInfo[] }>('/plugins')
    return resp.plugins
  },

  async enablePlugin(pluginName: string): Promise<{ enabled: boolean }> {
    return apiPost(`/plugins/${pluginName}/enable`)
  },

  async disablePlugin(pluginName: string): Promise<{ enabled: boolean }> {
    return apiPost(`/plugins/${pluginName}/disable`)
  },

  async reloadPlugins(): Promise<{ loaded: number }> {
    return apiPost('/plugins/reload')
  },

  async openwebuiDatasets(): Promise<any[]> {
    const resp = await apiGet<{ datasets: any[] }>('/openwebui/datasets')
    return resp.datasets
  },

  async openwebuiCheckpoints(): Promise<any[]> {
    const resp = await apiGet<{ checkpoints: any[] }>('/openwebui/checkpoints')
    return resp.checkpoints
  },

  async openwebuiStartTraining(datasetId: string, method: string = 'finetune'): Promise<any> {
    return apiPost('/openwebui/training/start', { dataset_id: datasetId, method })
  },

  async openwebuiStopTraining(): Promise<any> {
    return apiPost('/openwebui/training/stop')
  },

  async openwebuiTrainingStatus(): Promise<any> {
    return apiGet('/openwebui/training/status')
  },
}
