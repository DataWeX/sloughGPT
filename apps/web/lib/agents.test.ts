import {describe, it, expect} from 'vitest'
import {AGENTS, type AgentId} from './agents'

describe('agents', () => {
  it('has required agents', () => {
    const ids = Object.keys(AGENTS)
    expect(ids).toContain('general')
    expect(ids).toContain('coder')
    expect(ids).toContain('writer')
    expect(ids).toContain('researcher')
    expect(ids).toContain('analyst')
  })

  it('each agent has required fields', () => {
    for (const agent of Object.values(AGENTS)) {
      expect(agent.id).toBeTruthy()
      expect(agent.name).toBeTruthy()
      expect(agent.description).toBeTruthy()
      expect(agent.instructions).toBeTruthy()
    }
  })

  it('agent ids match their keys', () => {
    for (const [key, agent] of Object.entries(AGENTS)) {
      expect(agent.id).toBe(key)
    }
  })
})
