import { describe, expect, it, beforeEach } from 'vitest'

// Use dynamic import so we can reset store state between tests
import { useAppStore, getKnowledgeContext } from '../store'

describe('useAppStore settings', () => {
  beforeEach(() => {
    useAppStore.getState().updateSettings({
      apiUrl: 'http://localhost:8000',
      hfToken: '',
      defaultModel: 'gpt2',
      defaultTemp: 0.8,
      defaultMaxTokens: 200,
      theme: 'light',
      streaming: true,
      customContext: '',
    })
    useAppStore.getState().clearKnowledge()
  })

  it('starts with default settings', () => {
    const s = useAppStore.getState().settings
    expect(s.defaultModel).toBe('gpt2')
    expect(s.defaultTemp).toBe(0.8)
  })

  it('updateSettings merges partial', () => {
    useAppStore.getState().updateSettings({ defaultModel: 'qwen', defaultTemp: 0.5 })
    const s = useAppStore.getState().settings
    expect(s.defaultModel).toBe('qwen')
    expect(s.defaultTemp).toBe(0.5)
    expect(s.apiUrl).toBe('http://localhost:8000')
  })
})

describe('useAppStore knowledge', () => {
  beforeEach(() => {
    useAppStore.getState().clearKnowledge()
  })

  it('addKnowledge adds an item with generated id', () => {
    useAppStore.getState().addKnowledge('fact 1')
    const items = useAppStore.getState().injectedKnowledge
    expect(items).toHaveLength(1)
    expect(items[0].content).toBe('fact 1')
    expect(items[0].id).toMatch(/^know_\d+$/)
  })

  it('addKnowledge appends multiple items', () => {
    useAppStore.getState().addKnowledge('fact 1')
    useAppStore.getState().addKnowledge('fact 2')
    expect(useAppStore.getState().injectedKnowledge).toHaveLength(2)
  })

  it('removeKnowledge removes by id', () => {
    useAppStore.getState().addKnowledge('fact 1')
    const id = useAppStore.getState().injectedKnowledge[0].id
    useAppStore.getState().removeKnowledge(id)
    expect(useAppStore.getState().injectedKnowledge).toHaveLength(0)
  })

  it('clearKnowledge empties the list', () => {
    useAppStore.getState().addKnowledge('fact 1')
    useAppStore.getState().addKnowledge('fact 2')
    useAppStore.getState().clearKnowledge()
    expect(useAppStore.getState().injectedKnowledge).toEqual([])
  })
})

describe('getKnowledgeContext', () => {
  beforeEach(() => {
    useAppStore.getState().clearKnowledge()
    useAppStore.getState().updateSettings({ customContext: '' })
  })

  it('returns empty string when no knowledge', () => {
    expect(getKnowledgeContext()).toBe('')
  })

  it('includes customContext from settings', () => {
    useAppStore.getState().updateSettings({ customContext: 'Always be polite.' })
    const ctx = getKnowledgeContext()
    expect(ctx).toContain('Always be polite.')
    expect(ctx).toContain('[IMPORTANT KNOWLEDGE')
    expect(ctx).toContain('[/IMPORTANT KNOWLEDGE]')
  })

  it('includes injected knowledge items', () => {
    useAppStore.getState().addKnowledge('Paris is capital of France')
    const ctx = getKnowledgeContext()
    expect(ctx).toContain('Paris is capital of France')
  })
})
