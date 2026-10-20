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

/** Entry of GET /openwebui/datasets (router list_datasets). */
export interface OpenWebuiDataset {
  id: string
  name: string
  path: string
  workspace_id: string | null
}

/** Entry of GET /openwebui/checkpoints (router list_checkpoints). */
export interface OpenWebuiCheckpoint {
  id: string
  name: string
  path: string
}

/** Payload of GET /openwebui/training/status (router training_status). */
export interface OpenWebuiTrainingStatus {
  status: string
  progress: number
  current_epoch: number
  total_epochs: number
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

  async openwebuiDatasets(): Promise<OpenWebuiDataset[]> {
    const resp = await apiGet<{ datasets: OpenWebuiDataset[] }>('/openwebui/datasets')
    return resp.datasets
  },

  async openwebuiCheckpoints(): Promise<OpenWebuiCheckpoint[]> {
    const resp = await apiGet<{ checkpoints: OpenWebuiCheckpoint[] }>('/openwebui/checkpoints')
    return resp.checkpoints
  },

  async openwebuiStartTraining(datasetId: string, method: string = 'finetune'): Promise<{ status: string; dataset_id: string; method: string }> {
    return apiPost('/openwebui/training/start', { dataset_id: datasetId, method })
  },

  async openwebuiStopTraining(): Promise<{ status: string }> {
    return apiPost('/openwebui/training/stop')
  },

  async openwebuiTrainingStatus(): Promise<OpenWebuiTrainingStatus> {
    return apiGet('/openwebui/training/status')
  },
}
