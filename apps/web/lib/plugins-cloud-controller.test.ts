import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { pluginsCloudController } from './plugins-cloud-controller'

describe('pluginsCloudController.listCloudJobs', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /cloud-training/jobs', async () => {
    apiClient.apiGet.mockResolvedValue({ jobs: [{ job_id: 'j1', provider: 'aws', status: 'running', progress: 50 }] })
    const result = await pluginsCloudController.listCloudJobs()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/cloud-training/jobs?limit=10')
    expect(result).toHaveLength(1)
    expect(result[0].job_id).toBe('j1')
  })
})

describe('pluginsCloudController.submitCloudJob', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /cloud-training/submit', async () => {
    apiClient.apiPost.mockResolvedValue({ job_id: 'j2', status: 'queued' })
    const result = await pluginsCloudController.submitCloudJob('aws', 'ds-1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/cloud-training/submit', { provider: 'aws', dataset_id: 'ds-1' })
    expect(result.job_id).toBe('j2')
  })
})

describe('pluginsCloudController.cloudJobStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs job status', async () => {
    apiClient.apiGet.mockResolvedValue({ job_id: 'j1', status: 'completed', progress: 100 })
    const result = await pluginsCloudController.cloudJobStatus('j1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/cloud-training/j1/status')
    expect(result.status).toBe('completed')
  })
})

describe('pluginsCloudController.cancelCloudJob', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs cancel', async () => {
    apiClient.apiPost.mockResolvedValue({ cancelled: true })
    const result = await pluginsCloudController.cancelCloudJob('j1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/cloud-training/j1/cancel')
    expect(result.cancelled).toBe(true)
  })
})

describe('pluginsCloudController.listPlugins', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /plugins', async () => {
    apiClient.apiGet.mockResolvedValue({ plugins: [{ name: 'logger', enabled: true }] })
    const result = await pluginsCloudController.listPlugins()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/plugins')
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('logger')
  })
})

describe('pluginsCloudController.enablePlugin', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs enable', async () => {
    apiClient.apiPost.mockResolvedValue({ enabled: true })
    const result = await pluginsCloudController.enablePlugin('logger')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/plugins/logger/enable')
    expect(result.enabled).toBe(true)
  })
})

describe('pluginsCloudController.disablePlugin', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs disable', async () => {
    apiClient.apiPost.mockResolvedValue({ enabled: false })
    const result = await pluginsCloudController.disablePlugin('logger')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/plugins/logger/disable')
    expect(result.enabled).toBe(false)
  })
})

describe('pluginsCloudController.reloadPlugins', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs reload', async () => {
    apiClient.apiPost.mockResolvedValue({ loaded: 3 })
    const result = await pluginsCloudController.reloadPlugins()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/plugins/reload')
    expect(result.loaded).toBe(3)
  })
})

describe('pluginsCloudController.openwebuiDatasets', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /openwebui/datasets', async () => {
    apiClient.apiGet.mockResolvedValue({ datasets: [{ id: 'd1', name: 'My Dataset' }] })
    const result = await pluginsCloudController.openwebuiDatasets()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/openwebui/datasets')
    expect(result).toHaveLength(1)
  })
})

describe('pluginsCloudController.openwebuiCheckpoints', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /openwebui/checkpoints', async () => {
    apiClient.apiGet.mockResolvedValue({ checkpoints: [{ name: 'cp-1' }] })
    const result = await pluginsCloudController.openwebuiCheckpoints()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/openwebui/checkpoints')
    expect(result).toHaveLength(1)
  })
})

describe('pluginsCloudController.openwebuiStartTraining', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs start training', async () => {
    apiClient.apiPost.mockResolvedValue({ job_id: 'ow-1', status: 'started' })
    const result = await pluginsCloudController.openwebuiStartTraining('ds-1', 'lora')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/openwebui/training/start', { dataset_id: 'ds-1', method: 'lora' })
    expect(result.job_id).toBe('ow-1')
  })
})

describe('pluginsCloudController.openwebuiStopTraining', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs stop training', async () => {
    apiClient.apiPost.mockResolvedValue({ stopped: true })
    const result = await pluginsCloudController.openwebuiStopTraining()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/openwebui/training/stop')
    expect(result.stopped).toBe(true)
  })
})

describe('pluginsCloudController.openwebuiTrainingStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs training status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'idle', progress: 0 })
    const result = await pluginsCloudController.openwebuiTrainingStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/openwebui/training/status')
    expect(result.status).toBe('idle')
  })
})
