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

import { settingsController } from './settings-controller'

describe('settingsController.listTrainingPresets', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /settings/training/presets', async () => {
    apiClient.apiGet.mockResolvedValue({ presets: [{ name: 'quick-finetune' }] })
    const result = await settingsController.listTrainingPresets()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/presets')
    expect(result.presets).toHaveLength(1)
  })
})

describe('settingsController.applyTrainingPreset', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to apply preset', async () => {
    apiClient.apiPost.mockResolvedValue({ preset: 'quick-finetune', applied: {} })
    const result = await settingsController.applyTrainingPreset('quick-finetune')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/training/presets/quick-finetune/apply')
    expect(result.preset).toBe('quick-finetune')
  })
})

describe('settingsController.exportPresets', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /settings/training/presets/export', async () => {
    apiClient.apiGet.mockResolvedValue({ built_in: { 'quick-finetune': {} }, custom: {} })
    const result = await settingsController.exportPresets()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/presets/export')
    expect(result.built_in).toBeDefined()
    expect(result.custom).toBeDefined()
  })
})

describe('settingsController.importPresets', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs import data', async () => {
    apiClient.apiPost.mockResolvedValue({ imported: 2 })
    const data = { custom: { preset1: {}, preset2: {} } }
    const result = await settingsController.importPresets(data, true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/training/presets/import?overwrite=true', data)
    expect(result.imported).toBe(2)
  })
})

describe('settingsController.savePreset', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs preset data', async () => {
    const preset = { name: 'my-preset', model: 'gpt2', method: 'finetune' }
    apiClient.apiPost.mockResolvedValue(preset)
    const result = await settingsController.savePreset(preset)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/training/presets/save', preset)
    expect(result.name).toBe('my-preset')
  })
})

describe('settingsController.deletePreset', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs a preset', async () => {
    apiClient.apiDelete.mockResolvedValue({ deleted: 'my-preset' })
    const result = await settingsController.deletePreset('my-preset')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/settings/training/presets/my-preset')
    expect(result.deleted).toBe('my-preset')
  })
})

describe('settingsController.backupSettings', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /settings/backup', async () => {
    apiClient.apiGet.mockResolvedValue({ generation: {}, training: {} })
    const result = await settingsController.backupSettings()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/backup')
    expect(result.generation).toBeDefined()
  })
})

describe('settingsController.restoreSettings', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs restore data', async () => {
    apiClient.apiPost.mockResolvedValue({ restored_fields: 3 })
    const data = { training: { preferred_model: 'x' } }
    const result = await settingsController.restoreSettings(data)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/restore', data)
    expect(result.restored_fields).toBe(3)
  })
})

describe('settingsController.exportTrainingHistory', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs with format and limit', async () => {
    apiClient.apiGet.mockResolvedValue({ format: 'csv', content: 'a,b\n1,2', count: 1 })
    const result = await settingsController.exportTrainingHistory('csv', 100)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/history/export?format=csv&limit=100')
    expect(result.format).toBe('csv')
  })
})

describe('settingsController.filterTrainingRuns', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs with filter params', async () => {
    apiClient.apiGet.mockResolvedValue({ runs: [{ run_id: 'r1' }], total: 1 })
    const result = await settingsController.filterTrainingRuns({ model: 'gpt2', limit: 50 })
    expect(apiClient.apiGet).toHaveBeenCalled()
    expect(result.runs).toHaveLength(1)
  })
})

describe('settingsController.compareTrainingRuns', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs comparison', async () => {
    apiClient.apiGet.mockResolvedValue({ run_a: {}, run_b: {}, differences: {} })
    const result = await settingsController.compareTrainingRuns('run-a', 'run-b')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/compare?run_a=run-a&run_b=run-b')
    expect(result.differences).toBeDefined()
  })
})

describe('settingsController.getBatchTrainingStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs batch status', async () => {
    apiClient.apiGet.mockResolvedValue({ total: 5, completed: 3, failed: 1 })
    const result = await settingsController.getBatchTrainingStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/batch-status')
    expect(result.total).toBe(5)
  })
})

describe('settingsController.toggleBookmark', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs bookmark toggle', async () => {
    apiClient.apiPost.mockResolvedValue({ bookmarked: true })
    const result = await settingsController.toggleBookmark('run-1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/training/runs/run-1/bookmark')
    expect(result.bookmarked).toBe(true)
  })
})

describe('settingsController.getBookmarkedRuns', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs bookmarked runs', async () => {
    apiClient.apiGet.mockResolvedValue({ runs: [{ run_id: 'r1' }], count: 1 })
    const result = await settingsController.getBookmarkedRuns()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/bookmarks')
    expect(result.count).toBe(1)
  })
})

describe('settingsController.duplicateTrainingRun', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs duplicate', async () => {
    apiClient.apiPost.mockResolvedValue({ run_id: 'new-run' })
    const result = await settingsController.duplicateTrainingRun('old-run', 'new-run')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/training/runs/old-run/duplicate?new_run_id=new-run')
    expect(result.run_id).toBe('new-run')
  })
})

describe('settingsController.bulkDeleteRuns', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs bulk delete', async () => {
    apiClient.apiPost.mockResolvedValue({ deleted_count: 2, requested: 2 })
    const result = await settingsController.bulkDeleteRuns(['r1', 'r2'])
    expect(apiClient.apiPost).toHaveBeenCalled()
    expect(result.deleted_count).toBe(2)
  })
})

describe('settingsController.bulkAddTag', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs bulk tag', async () => {
    apiClient.apiPost.mockResolvedValue({ updated_count: 2, tag: 'best' })
    const result = await settingsController.bulkAddTag(['r1', 'r2'], 'best')
    expect(apiClient.apiPost).toHaveBeenCalled()
    expect(result.tag).toBe('best')
  })
})

describe('settingsController.bulkBookmark', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs bulk bookmark', async () => {
    apiClient.apiPost.mockResolvedValue({ updated_count: 2, bookmarked: true })
    const result = await settingsController.bulkBookmark(['r1', 'r2'], true)
    expect(apiClient.apiPost).toHaveBeenCalled()
    expect(result.bookmarked).toBe(true)
  })
})

describe('settingsController.addRunTag', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs tag', async () => {
    apiClient.apiPost.mockResolvedValue({ tags: ['best'] })
    const result = await settingsController.addRunTag('r1', 'best')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/training/runs/r1/tags?tag=best')
    expect(result.tags).toContain('best')
  })
})

describe('settingsController.removeRunTag', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs tag', async () => {
    apiClient.apiDelete.mockResolvedValue({ tags: [] })
    const result = await settingsController.removeRunTag('r1', 'best')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/settings/training/runs/r1/tags/best')
    expect(result.tags).toHaveLength(0)
  })
})

describe('settingsController.setRunNotes', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PUTs notes', async () => {
    apiClient.apiPut.mockResolvedValue({ notes: 'test note' })
    const result = await settingsController.setRunNotes('r1', 'test note')
    expect(apiClient.apiPut).toHaveBeenCalledWith('/settings/training/runs/r1/notes?notes=test%20note')
    expect(result.notes).toBe('test note')
  })
})

describe('settingsController.exportTrainingRun', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs export', async () => {
    apiClient.apiGet.mockResolvedValue({ content: '{}', format: 'json' })
    const result = await settingsController.exportTrainingRun('r1', 'json')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/runs/r1/export?format=json')
    expect(result.content).toBe('{}')
  })
})

describe('settingsController.generateModelCard', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs model card request', async () => {
    apiClient.apiPost.mockResolvedValue({ card: { model_name: 'test-model' }, markdown: '# test-model' })
    const result = await settingsController.generateModelCard('test-model', { base_model: 'gpt2' })
    expect(apiClient.apiPost).toHaveBeenCalled()
    expect(result.card.model_name).toBe('test-model')
  })
})

describe('settingsController.clearTrainingHistory', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs clear', async () => {
    apiClient.apiPost.mockResolvedValue({ cleared: 10 })
    const result = await settingsController.clearTrainingHistory()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/settings/training/history/clear')
    expect(result.cleared).toBe(10)
  })
})

describe('settingsController.getAutoTrainSettingsStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs auto-train status', async () => {
    apiClient.apiGet.mockResolvedValue({ enabled: true, threshold: 10 })
    const result = await settingsController.getAutoTrainSettingsStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/settings/training/auto-train/status')
    expect(result.enabled).toBe(true)
  })
})

describe('settingsController.updateAutoTrainSettingsConfig', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PATCHes auto-train config with threshold', async () => {
    apiClient.apiPatch.mockResolvedValue({ enabled: true, threshold: 20 })
    const result = await settingsController.updateAutoTrainSettingsConfig({ threshold: 20 })
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/settings/training/auto-train/config?threshold=20')
    expect(result.threshold).toBe(20)
  })

  it('PATCHes auto-train config with both params', async () => {
    apiClient.apiPatch.mockResolvedValue({ enabled: true, threshold: 30 })
    const result = await settingsController.updateAutoTrainSettingsConfig({ threshold: 30, interval_s: 300 })
    expect(apiClient.apiPatch).toHaveBeenCalledWith(expect.stringContaining('threshold=30'))
    expect(apiClient.apiPatch).toHaveBeenCalledWith(expect.stringContaining('interval_s=300'))
  })
})
