import { describe, expect, it, vi } from 'vitest'

// Persist rehydration reads localStorage synchronously at store creation, so the
// legacy payload must be seeded BEFORE the store module is loaded. A static
// import is hoisted above module-level statements, so a dynamic import is used.
const localStorageMock = (() => {
  let store = new Map<string, string>()
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => store.set(key, value),
    removeItem: (key: string) => store.delete(key),
    clear: () => store.clear(),
    get length() { return store.size },
    key: (i: number) => [...store.keys()][i] ?? null,
  }
})()

const LEGACY_PAYLOAD = JSON.stringify({
  state: {
    settings: {
      apiUrl: 'http://old-host:9999',
      hfToken: 'hf_legacy',
      theme: 'dark',
    },
    injectedKnowledge: [{ id: 'know_1', content: 'old fact', timestamp: 1 }],
  },
  version: 0,
})

localStorageMock.setItem('man-store', LEGACY_PAYLOAD)
vi.stubGlobal('localStorage', localStorageMock)

const { useAppStore } = await import('./store')

describe('persisted settings migration', () => {
  it('fills fields missing from a legacy persisted store with defaults', () => {
    const { settings } = useAppStore.getState()
    expect(settings.defaultTemp).toBe(0.7)
    expect(settings.defaultMaxTokens).toBe(300)
    expect(settings.defaultTopP).toBe(0.85)
    expect(settings.defaultTopK).toBe(40)
    expect(settings.collapsibleMessageLength).toBe(500)
    expect(settings.streaming).toBe(true)
  })

  it('keeps persisted values that still exist', () => {
    const { settings } = useAppStore.getState()
    expect(settings.apiUrl).toBe('http://old-host:9999')
    expect(settings.hfToken).toBe('hf_legacy')
    expect(settings.theme).toBe('dark')
  })

  it('preserves non-settings state', () => {
    const { injectedKnowledge } = useAppStore.getState()
    expect(injectedKnowledge).toHaveLength(1)
    expect(injectedKnowledge[0].content).toBe('old fact')
  })
})
