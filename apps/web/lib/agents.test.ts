import { describe, expect, it } from 'vitest'
import {
  AGENTS,
  getAgentName,
  getAgentDescription,
  getAgentSystemPrompt,
  getAgentCapabilities,
} from './agents'

describe('AGENTS', () => {
  it('contains all 5 agents', () => {
    expect(Object.keys(AGENTS)).toEqual(['general', 'coder', 'writer', 'researcher', 'analyst'])
  })

  it('each agent has required fields', () => {
    for (const agent of Object.values(AGENTS)) {
      expect(agent.id).toBeTruthy()
      expect(agent.name).toBeTruthy()
      expect(agent.description).toBeTruthy()
      expect(agent.instructions).toBeTruthy()
    }
  })
})

describe('getAgentName', () => {
  it('returns name for valid id', () => {
    expect(getAgentName('coder')).toBe('Coder')
  })

  it('returns fallback for unknown id', () => {
    expect(getAgentName('unknown')).toBe('General')
  })
})

describe('getAgentDescription', () => {
  it('returns description for valid id', () => {
    expect(getAgentDescription('researcher')).toContain('analysis')
  })

  it('returns empty string for unknown id', () => {
    expect(getAgentDescription('unknown')).toBe('')
  })
})

describe('getAgentSystemPrompt', () => {
  it('returns instructions for valid id', () => {
    expect(getAgentSystemPrompt('writer')).toContain('creative writing')
  })

  it('returns general instructions for unknown id', () => {
    const fallback = getAgentSystemPrompt('unknown')
    expect(fallback).toContain('helpful AI assistant')
  })
})

describe('getAgentCapabilities', () => {
  it('returns capabilities for valid id', () => {
    expect(getAgentCapabilities('analyst')).toContain('data-analysis')
    expect(getAgentCapabilities('analyst')).toContain('structured-reasoning')
  })

  it('returns empty array for unknown id', () => {
    expect(getAgentCapabilities('unknown')).toEqual([])
  })
})
