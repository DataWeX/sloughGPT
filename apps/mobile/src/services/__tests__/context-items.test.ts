import { contextItemsFromInspector, estimateTokens, type ContextInspector } from '../context-items'

describe('contextItemsFromInspector', () => {
  it('returns empty array for empty inspector', () => {
    expect(contextItemsFromInspector({})).toEqual([])
  })

  it('maps system_prompt to a system item', () => {
    const items = contextItemsFromInspector({ system_prompt: 'You are helpful.' })
    expect(items).toEqual([
      { type: 'system', label: 'System Prompt', content: 'You are helpful.', tokenCount: 4 },
    ])
  })

  it('groups session messages into user and assistant items', () => {
    const items = contextItemsFromInspector({
      session_messages: [
        { role: 'user', content: 'hi there' },
        { role: 'assistant', content: 'hello!' },
        { role: 'user', content: 'how are you' },
      ],
    })
    const user = items.find((i) => i.type === 'user')
    const assistant = items.find((i) => i.type === 'assistant')
    expect(user?.content).toContain('hi there')
    expect(user?.content).toContain('how are you')
    expect(assistant?.content).toContain('hello!')
  })

  it('maps semantic_keys to a knowledge item', () => {
    const items = contextItemsFromInspector({ semantic_keys: ['fav_color', 'city'] })
    const knowledge = items.find((i) => i.type === 'knowledge')
    expect(knowledge?.content).toContain('fav_color')
    expect(knowledge?.content).toContain('city')
  })

  it('maps working_memory to an agent item', () => {
    const items = contextItemsFromInspector({ working_memory: ['task: write tests'] })
    const agent = items.find((i) => i.type === 'agent')
    expect(agent?.content).toContain('task: write tests')
  })

  it('skips empty sections', () => {
    const items = contextItemsFromInspector({
      system_prompt: '',
      session_messages: [],
      semantic_keys: [],
      working_memory: [],
      episodic_count: 0,
    })
    expect(items).toEqual([])
  })

  it('produces unique labels for grouping', () => {
    const items = contextItemsFromInspector({
      system_prompt: 'sys',
      session_messages: [{ role: 'user', content: 'a' }],
      semantic_keys: ['k'],
      working_memory: ['w'],
    })
    const labels = items.map((i) => i.label)
    expect(new Set(labels).size).toBe(labels.length)
  })
})

describe('estimateTokens', () => {
  it('uses provided totalTokens', () => {
    expect(estimateTokens([], 42)).toBe(42)
  })

  it('sums item tokenCounts', () => {
    expect(
      estimateTokens([
        { type: 'system', label: 'a', content: 'x', tokenCount: 3 },
        { type: 'user', label: 'b', content: 'y', tokenCount: 4 },
      ]),
    ).toBe(7)
  })

  it('falls back to length/4 when no tokenCount', () => {
    expect(estimateTokens([{ type: 'system', label: 'a', content: 'a'.repeat(40) }])).toBe(10)
  })
})
