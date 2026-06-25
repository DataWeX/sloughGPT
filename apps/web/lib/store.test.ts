import { beforeEach, describe, expect, it, vi } from 'vitest'

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

vi.stubGlobal('localStorage', localStorageMock)

import { useAppStore, getKnowledgeContext } from './store'

beforeEach(() => {
  useAppStore.setState({ settings: useAppStore.getInitialState().settings, injectedKnowledge: [] })
})

describe('useAppStore', () => {
  it('has default settings', () => {
    const { settings } = useAppStore.getState()
    expect(settings.apiUrl).toBe('http://localhost:8000')
    expect(settings.defaultModel).toBe('gpt2')
    expect(settings.defaultTemp).toBe(0.8)
    expect(settings.theme).toBe('light')
    expect(settings.streaming).toBe(true)
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
      expect(settings.defaultModel).toBe('gpt2')
    })

    it('preserves unchanged fields', () => {
      useAppStore.getState().updateSettings({ defaultModel: 'llama' })
      const { settings } = useAppStore.getState()
      expect(settings.defaultModel).toBe('llama')
      expect(settings.theme).toBe('light')
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
  })

  describe('clearKnowledge', () => {
    it('removes all items', () => {
      useAppStore.getState().addKnowledge('fact 1')
      useAppStore.getState().addKnowledge('fact 2')
      useAppStore.getState().clearKnowledge()
      expect(useAppStore.getState().injectedKnowledge).toEqual([])
    })
  })
})

describe('getKnowledgeContext', () => {
  it('returns empty string when no knowledge', () => {
    const result = getKnowledgeContext()
    expect(result).toBe('')
  })

  it('includes customContext', () => {
    useAppStore.getState().updateSettings({ customContext: 'You are a helpful assistant.' })
    const result = getKnowledgeContext()
    expect(result).toContain('You are a helpful assistant.')
    expect(result).toMatch(/^\n\n\[IMPORTANT KNOWLEDGE/)
  })

  it('includes injected knowledge items', () => {
    useAppStore.getState().addKnowledge('Paris is the capital of France')
    const result = getKnowledgeContext()
    expect(result).toContain('Paris is the capital of France')
  })

  it('combines customContext and injected knowledge', () => {
    useAppStore.getState().updateSettings({ customContext: 'Be concise.' })
    useAppStore.getState().addKnowledge('The sky is blue')
    const result = getKnowledgeContext()
    expect(result).toContain('Be concise.')
    expect(result).toContain('The sky is blue')
  })
})
