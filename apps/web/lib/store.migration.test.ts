import { describe, expect, it, vi } from 'vitest'

const chatDBMock = {
  getKV: vi.fn().mockResolvedValue(undefined),
  setKV: vi.fn().mockResolvedValue(undefined),
  deleteKV: vi.fn().mockResolvedValue(undefined),
}

vi.mock('@/lib/db', () => ({
  chatDB: chatDBMock,
}))

const { useAppStore, initStore, DEFAULT_SETTINGS } = await import('./store')

describe('MogDB-backed settings initialization', () => {
  beforeEach(() => {
    useAppStore.setState({ settings: DEFAULT_SETTINGS, injectedKnowledge: [] })
    chatDBMock.getKV.mockClear()
    chatDBMock.setKV.mockClear()
    chatDBMock.deleteKV.mockClear()
  })

  it('starts with defaults when MogDB has no stored settings', async () => {
    chatDBMock.getKV.mockResolvedValueOnce(undefined)
    await initStore()
    const { settings } = useAppStore.getState()
    expect(settings.defaultTemp).toBe(0.7)
    expect(settings.defaultMaxTokens).toBe(300)
    expect(settings.streaming).toBe(true)
  })

  it('overrides defaults with MogDB-stored values', async () => {
    chatDBMock.getKV
      .mockResolvedValueOnce({ defaultTemp: 0.5, hfToken: 'hf_test', theme: 'dark' })
      .mockResolvedValueOnce(undefined)

    await initStore()
    const { settings } = useAppStore.getState()
    expect(settings.defaultTemp).toBe(0.5)
    expect(settings.hfToken).toBe('hf_test')
    expect(settings.theme).toBe('dark')
    expect(settings.defaultMaxTokens).toBe(300)
  })

  it('preserves non-settings state when loading settings', async () => {
    useAppStore.setState({ injectedKnowledge: [{ id: 'know_1', content: 'local fact', timestamp: 1 }] })
    chatDBMock.getKV
      .mockResolvedValueOnce({ defaultTemp: 0.5 })
      .mockResolvedValueOnce(undefined)

    await initStore()
    const { injectedKnowledge } = useAppStore.getState()
    expect(injectedKnowledge).toHaveLength(1)
    expect(injectedKnowledge[0].content).toBe('local fact')
  })

  it('loads injected knowledge from MogDB', async () => {
    const storedKnowledge = [
      { id: 'know_1', content: 'fact from db', timestamp: 1000 },
      { id: 'know_2', content: 'another fact', timestamp: 2000 },
    ]
    chatDBMock.getKV
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce(storedKnowledge)

    await initStore()
    const { injectedKnowledge } = useAppStore.getState()
    expect(injectedKnowledge).toHaveLength(2)
    expect(injectedKnowledge[0].content).toBe('fact from db')
    expect(injectedKnowledge[1].content).toBe('another fact')
  })

  it('handles MogDB errors gracefully', async () => {
    chatDBMock.getKV.mockRejectedValueOnce(new Error('MogDB down'))
    await initStore()
    const { settings } = useAppStore.getState()
    expect(settings.defaultTemp).toBe(0.7)
  })
})
