import { describe, it, expect } from 'vitest'
import {
  knowledgeSchema,
  agentSchema,
  agentExecuteSchema,
  orchestrateSchema,
  settingsSchema,
  quickNoteSchema,
  systemPromptSchema,
  snapshotSchema,
  renameSchema,
} from './validation-schemas'

describe('knowledgeSchema', () => {
  it('rejects empty content', () => {
    const result = knowledgeSchema.safeParse({ content: '', topic: 'test' })
    expect(result.success).toBe(false)
  })

  it('accepts valid content', () => {
    const result = knowledgeSchema.safeParse({ content: 'Paris is the capital of France', topic: 'geography' })
    expect(result.success).toBe(true)
  })

  it('allows empty topic', () => {
    const result = knowledgeSchema.safeParse({ content: 'test', topic: '' })
    expect(result.success).toBe(true)
  })

  it('rejects content over 10000 chars', () => {
    const result = knowledgeSchema.safeParse({ content: 'x'.repeat(10001), topic: 'test' })
    expect(result.success).toBe(false)
  })

  it('rejects topic over 100 chars', () => {
    const result = knowledgeSchema.safeParse({ content: 'test', topic: 'x'.repeat(101) })
    expect(result.success).toBe(false)
  })
})

describe('agentSchema', () => {
  it('rejects empty name', () => {
    const result = agentSchema.safeParse({ name: '', description: '', instructions: '' })
    expect(result.success).toBe(false)
  })

  it('accepts valid agent', () => {
    const result = agentSchema.safeParse({ name: 'Researcher', description: 'Finds info', instructions: 'Be helpful' })
    expect(result.success).toBe(true)
  })

  it('rejects name over 50 chars', () => {
    const result = agentSchema.safeParse({ name: 'x'.repeat(51), description: '', instructions: '' })
    expect(result.success).toBe(false)
  })
})

describe('agentExecuteSchema', () => {
  it('rejects empty prompt', () => {
    const result = agentExecuteSchema.safeParse({ prompt: '' })
    expect(result.success).toBe(false)
  })

  it('accepts valid prompt', () => {
    const result = agentExecuteSchema.safeParse({ prompt: 'What is AI?' })
    expect(result.success).toBe(true)
  })
})

describe('orchestrateSchema', () => {
  it('rejects empty goal', () => {
    const result = orchestrateSchema.safeParse({ goal: '', context: '' })
    expect(result.success).toBe(false)
  })

  it('accepts valid goal', () => {
    const result = orchestrateSchema.safeParse({ goal: 'Research AI trends', context: '' })
    expect(result.success).toBe(true)
  })
})

describe('settingsSchema', () => {
  it('rejects invalid URL', () => {
    const result = settingsSchema.safeParse({ apiUrl: 'not-a-url', hfToken: '', customContext: '' })
    expect(result.success).toBe(false)
  })

  it('accepts valid URL', () => {
    const result = settingsSchema.safeParse({ apiUrl: 'http://localhost:8000', hfToken: '', customContext: '' })
    expect(result.success).toBe(true)
  })

  it('accepts empty values', () => {
    const result = settingsSchema.safeParse({ apiUrl: '', hfToken: '', customContext: '' })
    expect(result.success).toBe(true)
  })
})

describe('quickNoteSchema', () => {
  it('rejects empty text', () => {
    const result = quickNoteSchema.safeParse({ text: '' })
    expect(result.success).toBe(false)
  })

  it('accepts valid note', () => {
    const result = quickNoteSchema.safeParse({ text: 'Remember this' })
    expect(result.success).toBe(true)
  })
})

describe('systemPromptSchema', () => {
  it('rejects empty prompt', () => {
    const result = systemPromptSchema.safeParse({ prompt: '', presetName: 'test' })
    expect(result.success).toBe(false)
  })

  it('rejects empty preset name', () => {
    const result = systemPromptSchema.safeParse({ prompt: 'Be helpful', presetName: '' })
    expect(result.success).toBe(false)
  })
})

describe('snapshotSchema', () => {
  it('rejects empty name', () => {
    const result = snapshotSchema.safeParse({ name: '' })
    expect(result.success).toBe(false)
  })

  it('accepts valid name', () => {
    const result = snapshotSchema.safeParse({ name: 'My Snapshot' })
    expect(result.success).toBe(true)
  })
})

describe('renameSchema', () => {
  it('rejects empty name', () => {
    const result = renameSchema.safeParse({ name: '' })
    expect(result.success).toBe(false)
  })

  it('accepts valid name', () => {
    const result = renameSchema.safeParse({ name: 'New Name' })
    expect(result.success).toBe(true)
  })

  it('rejects name over 200 chars', () => {
    const result = renameSchema.safeParse({ name: 'x'.repeat(201) })
    expect(result.success).toBe(false)
  })
})
