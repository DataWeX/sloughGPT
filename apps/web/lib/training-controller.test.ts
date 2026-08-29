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

import { trainingJobsController } from './training-controller'

describe('trainingJobsController.startAutoTrain', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /auto-train/start with params', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started', teacher: 'gpt2', student: 'lstm' })

    const result = await trainingJobsController.startAutoTrain({ teacher_model: 'gpt2' })
    expect(result.status).toBe('started')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/auto-train/start', { teacher_model: 'gpt2' })
  })

  it('sends null body when no params', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started' })
    await trainingJobsController.startAutoTrain()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/auto-train/start', null)
  })
})

describe('trainingJobsController.stopAutoTrain', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /auto-train/stop', async () => {
    apiClient.apiPost.mockResolvedValue(undefined)
    await trainingJobsController.stopAutoTrain()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/auto-train/stop')
  })
})

describe('trainingJobsController.loadAdapter', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/load-adapter with adapter_path and merge', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'loaded', rank: 8, merged: false })
    const result = await trainingJobsController.loadAdapter('/path/to/adapter.npz', false)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/load-adapter', { adapter_path: '/path/to/adapter.npz', merge: false })
    expect(result.status).toBe('loaded')
  })
})

describe('trainingJobsController.startTurboTrain', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /auto-train/start-turbo', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'done', final_loss: 0.5 })
    const result = await trainingJobsController.startTurboTrain({ epochs: 5 })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/auto-train/start-turbo', { epochs: 5 })
    expect(result.status).toBe('done')
  })
})

describe('trainingJobsController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /training/jobs and returns array', async () => {
    apiClient.apiGet.mockResolvedValue([{ id: 'j1', name: 'test', status: 'done', progress: 100, created_at: '' }])
    const result = await trainingJobsController.list()
    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('j1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/jobs')
  })

  it('handles {jobs: [...]} response shape', async () => {
    apiClient.apiGet.mockResolvedValue({ jobs: [{ id: 'j2', name: 'test', status: 'running', progress: 50, created_at: '' }] })
    const result = await trainingJobsController.list()
    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('j2')
  })
})

describe('trainingJobsController.listCheckpoints', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /auto-train/checkpoints', async () => {
    apiClient.apiGet.mockResolvedValue([{ name: 'v1', soul: 'friendly' }])
    const result = await trainingJobsController.listCheckpoints()
    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/auto-train/checkpoints')
  })
})

describe('trainingJobsController.listBuilds', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /training/builds', async () => {
    apiClient.apiGet.mockResolvedValue({ builds: [{ name: 'b1', build_type: 'hf-finetune' }] })
    const result = await trainingJobsController.listBuilds()
    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/builds')
  })
})

describe('trainingJobsController.get', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /training/jobs/{id}', async () => {
    apiClient.apiGet.mockResolvedValue({ id: 'j1', name: 'test', status: 'done', progress: 100, created_at: '' })
    const result = await trainingJobsController.get('j1')
    expect(result?.id).toBe('j1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/jobs/j1')
  })

  it('returns null on 404', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('404 not found'))
    const result = await trainingJobsController.get('missing')
    expect(result).toBeNull()
  })
})

describe('trainingJobsController.create', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/start with params', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started', job_id: 'j1' })
    const result = await trainingJobsController.create({
      name: 'test',
      model: 'gpt2',
      dataset: 'shakespeare',
    })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/start', {
      name: 'test',
      model: 'gpt2',
      dataset: 'shakespeare',
      epochs: undefined,
      batch_size: undefined,
      learning_rate: undefined,
      device: undefined,
      use_lora: undefined,
      lora_rank: undefined,
    })
    expect(result.status).toBe('started')
  })
})

describe('trainingJobsController.startLoraFinetune', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/lora-finetune', async () => {
    apiClient.apiPost.mockResolvedValue({ job_id: 'lora1', status: 'queued', message: 'ok' })
    const result = await trainingJobsController.startLoraFinetune({
      model_path: 'models/gpt2.slnc',
      dataset: 'shakespeare',
    })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/lora-finetune', {
      model_path: 'models/gpt2.slnc',
      dataset: 'shakespeare',
      name: undefined,
      rank: undefined,
      alpha: undefined,
      dropout: undefined,
      target_modules: undefined,
      epochs: undefined,
      batch_size: undefined,
      block_size: undefined,
      learning_rate: undefined,
      warmup_steps: undefined,
      weight_decay: undefined,
      grad_clip: undefined,
      grad_accumulation_steps: undefined,
      log_interval: undefined,
      output_dir: undefined,
      adapter_name: undefined,
    })
    expect(result.job_id).toBe('lora1')
  })
})

describe('trainingJobsController.startQuick', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/quick', async () => {
    apiClient.apiPost.mockResolvedValue({ job_id: 'q1', status: 'ok', config: {}, explanation: 'done' })
    const result = await trainingJobsController.startQuick({ dataset: 'shakespeare' })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/quick', {
      dataset: 'shakespeare',
      name: undefined,
      model: undefined,
    })
    expect(result.job_id).toBe('q1')
  })
})

describe('trainingJobsController.getSummary', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /training/jobs/{id}/summary', async () => {
    apiClient.apiGet.mockResolvedValue({ job_id: 'j1', summary: 'test', status: 'done', model: 'gpt2', dataset: 'sh' })
    const result = await trainingJobsController.getSummary('j1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/jobs/j1/summary')
    expect(result.job_id).toBe('j1')
  })
})

describe('trainingJobsController.startVisualTrain', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/visual-start', async () => {
    apiClient.apiPost.mockResolvedValue({ job_id: 'visual1', status: 'ok', message: 'started' })
    await trainingJobsController.startVisualTrain({ dataset: 'images' })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/visual-start', { dataset: 'images' })
  })
})

describe('trainingJobsController.stop', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/jobs/{id}/stop', async () => {
    apiClient.apiPost.mockResolvedValue(undefined)
    await trainingJobsController.stop('j1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/jobs/j1/stop')
  })
})

describe('trainingJobsController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /training/jobs/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)
    await trainingJobsController.delete('j1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/training/jobs/j1')
  })
})

describe('trainingJobsController.recoverable', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /recovery/recoverable', async () => {
    apiClient.apiGet.mockResolvedValue({ jobs: [{ id: 'r1', name: 'crashed', failed_at: '' }] })
    const result = await trainingJobsController.recoverable()
    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/recovery/recoverable')
  })
})

describe('trainingJobsController.recover', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /recovery/recover/{id}', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'recovered' })
    const result = await trainingJobsController.recover('j1')
    expect(result.status).toBe('recovered')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/recovery/recover/j1')
  })
})

describe('trainingJobsController.webhooks', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('listWebhooks GETs /training/webhooks', async () => {
    apiClient.apiGet.mockResolvedValue({ webhooks: [{ id: 'w1', url: 'http://hook', events: ['done'] }] })
    const result = await trainingJobsController.listWebhooks()
    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/webhooks')
  })

  it('createWebhook POSTs to /training/webhooks', async () => {
    apiClient.apiPost.mockResolvedValue({ id: 'w2', url: 'http://new', events: ['start'] })
    await trainingJobsController.createWebhook('http://new', ['start'])
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/webhooks', { url: 'http://new', events: ['start'] })
  })

  it('deleteWebhook DELETEs /training/webhooks/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)
    await trainingJobsController.deleteWebhook('w1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/training/webhooks/w1')
  })

  it('webhookStats GETs /training/webhooks/stats', async () => {
    apiClient.apiGet.mockResolvedValue({ total_webhooks: 5, success_rate: 0.9, total_deliveries: 10, successful_deliveries: 9, failed_deliveries: 1, active_webhooks: 3, pending_retries: 0, dead_letters: 0 })
    const result = await trainingJobsController.webhookStats()
    expect(result.total_webhooks).toBe(5)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/webhooks/stats')
  })
})

describe('trainingJobsController.recovery', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('getRecoveryStats GETs /recovery/stats', async () => {
    apiClient.apiGet.mockResolvedValue({ recovered: 3, failed: 1 })
    const result = await trainingJobsController.getRecoveryStats()
    expect(result.recovered).toBe(3)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/recovery/stats')
  })

  it('abandon DELETEs /recovery/abandon/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue({ message: 'done' })
    const result = await trainingJobsController.abandon('j1')
    expect(result.message).toBe('done')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/recovery/abandon/j1')
  })
})

describe('trainingJobsController.testWebhook', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/webhooks/test', async () => {
    apiClient.apiPost.mockResolvedValue({ ok: true })
    await trainingJobsController.testWebhook('http://hook')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/webhooks/test', { url: 'http://hook' })
  })
})

describe('trainingJobsController.getStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /training/status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'idle' })
    const result = await trainingJobsController.getStatus()
    expect(result.status).toBe('idle')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/status')
  })
})

describe('trainingJobsController.exportFeedbackPairs', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/export-text', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', count: 10 })
    const result = await trainingJobsController.exportFeedbackPairs(0.5, 100)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/export-text', { min_quality: 0.5, target_count: 100 })
    expect(result.count).toBe(10)
  })
})

describe('trainingJobsController.loadCheckpoint', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /auto-train/checkpoints/{name}/load', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true })
    const result = await trainingJobsController.loadCheckpoint('my-checkpoint')
    expect(result.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/auto-train/checkpoints/my-checkpoint/load')
  })
})

describe('trainingJobsController.deleteCheckpoint', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /auto-train/checkpoints/{name}', async () => {
    apiClient.apiDelete.mockResolvedValue({ success: true })
    const result = await trainingJobsController.deleteCheckpoint('old-checkpoint')
    expect(result.success).toBe(true)
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/auto-train/checkpoints/old-checkpoint')
  })
})

describe('trainingJobsController.deleteCheckpointsBatch', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('deletes multiple checkpoints in parallel', async () => {
    apiClient.apiDelete.mockResolvedValue({ success: true })
    const result = await trainingJobsController.deleteCheckpointsBatch(['cp1', 'cp2', 'cp3'])
    expect(result.deleted).toBe(3)
    expect(apiClient.apiDelete).toHaveBeenCalledTimes(3)
  })

  it('counts partial successes', async () => {
    apiClient.apiDelete
      .mockResolvedValueOnce({ success: true })
      .mockRejectedValueOnce(new Error('not found'))
      .mockResolvedValueOnce({ success: true })
    const result = await trainingJobsController.deleteCheckpointsBatch(['cp1', 'cp2', 'cp3'])
    expect(result.deleted).toBe(2)
  })
})

describe('trainingJobsController.listFineTuned', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('returns models from /training/finetuned-models', async () => {
    apiClient.apiGet.mockResolvedValue({
      models: [{ name: 'gpt2_dataset_1', model: 'gpt2', dataset: 'dataset_1', size_mb: 1.2, model_path: '/tmp/x' }],
    })
    const result = await trainingJobsController.listFineTuned()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/finetuned-models')
    expect(result).toHaveLength(1)
    expect(result[0].model).toBe('gpt2')
  })

  it('accepts a bare array response', async () => {
    apiClient.apiGet.mockResolvedValue([{ name: 'gpt2_dataset_1', model: 'gpt2', dataset: '', size_mb: 0, model_path: '/tmp/x' }])
    const result = await trainingJobsController.listFineTuned()
    expect(result).toHaveLength(1)
  })

  it('propagates errors', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('boom'))
    await expect(trainingJobsController.listFineTuned()).rejects.toThrow('boom')
  })
})

describe('trainingJobsController.loadFineTuned', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/finetuned-models/{name}/load', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'loaded', name: 'gpt2_dataset_1', model_path: '/tmp/x', model_id: 'gpt2' })
    const result = await trainingJobsController.loadFineTuned('gpt2_dataset_1')
    expect(result.model_id).toBe('gpt2')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/finetuned-models/gpt2_dataset_1/load')
  })

  it('URL-encodes special characters in name', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'loaded', name: 'a b', model_path: '/tmp/x' })
    await trainingJobsController.loadFineTuned('a b')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/finetuned-models/a%20b/load')
  })
})

describe('trainingJobsController.deleteFineTuned', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /training/finetuned-models/{name}', async () => {
    apiClient.apiDelete.mockResolvedValue({ status: 'deleted', name: 'gpt2_dataset_1' })
    const result = await trainingJobsController.deleteFineTuned('gpt2_dataset_1')
    expect(result.status).toBe('deleted')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/training/finetuned-models/gpt2_dataset_1')
  })
})
