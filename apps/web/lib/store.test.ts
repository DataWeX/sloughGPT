import { beforeEach, describe, expect, it, vi } from 'vitest'

const { chatDBMock, settingsControllerMock } = vi.hoisted(() => {
  const chatDBMock = {
    getKV: vi.fn().mockResolvedValue(undefined),
    setKV: vi.fn().mockResolvedValue(undefined),
    deleteKV: vi.fn().mockResolvedValue(undefined),
  }
  const settingsControllerMock = {
    getGeneration: vi.fn(),
    updateGeneration: vi.fn().mockResolvedValue({}),
    getAll: vi.fn(),
  }
  return { chatDBMock, settingsControllerMock }
})

vi.mock('@/lib/db', () => ({
  chatDB: chatDBMock,
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: settingsControllerMock,
}))

import { useAppStore, DEFAULT_SETTINGS } from './store'

beforeEach(() => {
  useAppStore.setState({ settings: DEFAULT_SETTINGS, injectedKnowledge: [] })
  chatDBMock.getKV.mockClear()
  chatDBMock.setKV.mockClear()
  chatDBMock.deleteKV.mockClear()
})

describe('useAppStore', () => {
  it('has default settings', () => {
    const { settings } = useAppStore.getState()
    expect(settings.apiUrl).toBe('http://localhost:8000')
    expect(settings.defaultTemp).toBe(0.7)
    expect(settings.theme).toBe('light')
    expect(settings.streaming).toBe(true)
  })

  it('has default connection-banner timing flags', () => {
    const { settings } = useAppStore.getState()
    expect(settings.bannerGraceMs).toBe(5_000)
    expect(settings.bannerHoldMs).toBe(4_000)
  })

  it('starts with empty knowledge', () => {
    const { injectedKnowledge } = useAppStore.getState()
    expect(injectedKnowledge).toEqual([])
  })

  describe('updateSettings', () => {
    it('merges settings', () => {
      useAppStore.getState().updateSettings({ defaultTemp: 0.5, theme: 'dark' })
      const { settings } = useAppStore.getState()
      expect(settings.defaultTemp).toBe(0.5)
      expect(settings.theme).toBe('dark')
    })

    it('preserves unchanged fields', () => {
      useAppStore.getState().updateSettings({ defaultMaxTokens: 500 })
      const { settings } = useAppStore.getState()
      expect(settings.defaultMaxTokens).toBe(500)
      expect(settings.theme).toBe('light')
    })

    it('persists settings to MogDB', () => {
      vi.useFakeTimers()
      useAppStore.getState().updateSettings({ defaultTemp: 0.5 })
      vi.advanceTimersByTime(500)
      expect(chatDBMock.setKV).toHaveBeenCalledWith(
        'app-settings',
        expect.objectContaining({ defaultTemp: 0.5 }),
      )
      vi.useRealTimers()
    })
  })

  // Card 20261102_085248: the health-driven fetch re-sends the same values
  // every ~3 s. In steady state nothing changed, so nothing may leave — no
  // backend PATCH, no localStorage flush, no state churn.
  describe('updateSettings identical-payload guard', () => {
    it('never pushes or persists when every key is unchanged', () => {
      vi.useFakeTimers()
      try {
        chatDBMock.setKV.mockClear()
        settingsControllerMock.updateGeneration.mockClear()
        for (let i = 0; i < 5; i++) {
          useAppStore.getState().updateSettings({
            defaultTemp: DEFAULT_SETTINGS.defaultTemp,
            defaultMaxTokens: DEFAULT_SETTINGS.defaultMaxTokens,
          })
          vi.advanceTimersByTime(3_000)
        }
        expect(settingsControllerMock.updateGeneration).not.toHaveBeenCalled()
        expect(chatDBMock.setKV).not.toHaveBeenCalled()
      } finally {
        vi.useRealTimers()
      }
    })

    it('pushes only the keys that actually changed', () => {
      settingsControllerMock.updateGeneration.mockClear()
      useAppStore.getState().updateSettings({
        defaultTemp: DEFAULT_SETTINGS.defaultTemp, // unchanged
        defaultMaxTokens: 500, // changed
      })
      expect(settingsControllerMock.updateGeneration).toHaveBeenCalledTimes(1)
      expect(settingsControllerMock.updateGeneration).toHaveBeenCalledWith({
        max_new_tokens: 500,
      })
      expect(useAppStore.getState().settings.defaultMaxTokens).toBe(500)
    })

    it('a real change still round-trips to the backend', () => {
      settingsControllerMock.updateGeneration.mockClear()
      useAppStore.getState().updateSettings({ defaultTemp: 0.9 })
      expect(settingsControllerMock.updateGeneration).toHaveBeenCalledWith({ temperature: 0.9 })
      expect(useAppStore.getState().settings.defaultTemp).toBe(0.9)
    })
  })

  describe('addKnowledge', () => {
    it('adds an item with id and timestamp', () => {
      useAppStore.getState().addKnowledge('test fact')
      const { injectedKnowledge } = useAppStore.getState()
      expect(injectedKnowledge).toHaveLength(1)
      expect(injectedKnowledge[0].content).toBe('test fact')
      expect(injectedKnowledge[0].id).toMatch(/^know_\d+$/)
      expect(injectedKnowledge[0].timestamp).toBeGreaterThan(0)
    })

    it('appends to existing knowledge', () => {
      useAppStore.getState().addKnowledge('fact 1')
      useAppStore.getState().addKnowledge('fact 2')
      expect(useAppStore.getState().injectedKnowledge).toHaveLength(2)
    })

    it('persists knowledge to MogDB', () => {
      useAppStore.getState().addKnowledge('test fact')
      expect(chatDBMock.setKV).toHaveBeenCalledWith(
        'app-knowledge',
        expect.arrayContaining([expect.objectContaining({ content: 'test fact' })]),
      )
    })
  })

  describe('removeKnowledge', () => {
    it('removes by id', () => {
      vi.useFakeTimers()
      useAppStore.getState().addKnowledge('keep me')
      vi.advanceTimersByTime(1)
      useAppStore.getState().addKnowledge('remove me')
      vi.useRealTimers()
      const items = useAppStore.getState().injectedKnowledge
      const toRemove = items[1].id
      useAppStore.getState().removeKnowledge(toRemove)
      expect(useAppStore.getState().injectedKnowledge).toHaveLength(1)
      expect(useAppStore.getState().injectedKnowledge[0].content).toBe('keep me')
    })

    it('handles unknown id gracefully', () => {
      useAppStore.getState().addKnowledge('fact')
      useAppStore.getState().removeKnowledge('nonexistent')
      expect(useAppStore.getState().injectedKnowledge).toHaveLength(1)
    })

    it('persists removal to MogDB', () => {
      useAppStore.getState().addKnowledge('fact 1')
      useAppStore.getState().addKnowledge('fact 2')
      const idToRemove = useAppStore.getState().injectedKnowledge[0].id
      useAppStore.getState().removeKnowledge(idToRemove)
      expect(chatDBMock.setKV).toHaveBeenCalledWith(
        'app-knowledge',
        expect.arrayContaining([expect.objectContaining({ content: 'fact 2' })]),
      )
    })
  })

  describe('clearKnowledge', () => {
    it('removes all items', () => {
      useAppStore.getState().addKnowledge('fact 1')
      useAppStore.getState().addKnowledge('fact 2')
      useAppStore.getState().clearKnowledge()
      expect(useAppStore.getState().injectedKnowledge).toEqual([])
    })

    it('clears knowledge from MogDB', () => {
      useAppStore.getState().addKnowledge('fact 1')
      useAppStore.getState().clearKnowledge()
      expect(chatDBMock.deleteKV).toHaveBeenCalledWith('app-knowledge')
    })
  })
})
