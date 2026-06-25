// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

import { AgentsTab } from './AgentsTab'

describe('AgentsTab', () => {
  const onSelectAgent = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('shows empty state when no agents', () => {
    render(<AgentsTab agents={[]} currentAgent={null} onSelectAgent={onSelectAgent} />)
    expect(screen.getByText(/No agents configured/)).toBeDefined()
  })

  it('renders Default (no agent) button', () => {
    render(
      <AgentsTab
        agents={[{ id: 'a1', name: 'Helper', description: '', instructions: '' }]}
        currentAgent={null}
        onSelectAgent={onSelectAgent}
      />
    )
    expect(screen.getByText('Direct chat')).toBeDefined()
  })

  it('renders agent names', () => {
    render(
      <AgentsTab
        agents={[
          { id: 'a1', name: 'Helper', description: 'Helps with tasks', instructions: '' },
          { id: 'a2', name: 'Critic', description: 'Reviews code', instructions: '' },
        ]}
        currentAgent={null}
        onSelectAgent={onSelectAgent}
      />
    )
    expect(screen.getByText('Helper')).toBeDefined()
    expect(screen.getByText('Critic')).toBeDefined()
  })

  it('shows agent descriptions', () => {
    render(
      <AgentsTab
        agents={[{ id: 'a1', name: 'Helper', description: 'Helps with tasks', instructions: '' }]}
        currentAgent={null}
        onSelectAgent={onSelectAgent}
      />
    )
    expect(screen.getByText('Helps with tasks')).toBeDefined()
  })

  it('calls onSelectAgent(null) when Default clicked', () => {
    render(
      <AgentsTab
        agents={[{ id: 'a1', name: 'Helper', description: '', instructions: '' }]}
        currentAgent={{ id: 'a1', name: 'Helper', description: '', instructions: '' }}
        onSelectAgent={onSelectAgent}
      />
    )
    fireEvent.click(screen.getByText('Direct chat'))
    expect(onSelectAgent).toHaveBeenCalledWith(null)
  })

  it('calls onSelectAgent with agent when agent clicked', () => {
    const agent = { id: 'a1', name: 'Helper', description: '', instructions: '' }
    render(
      <AgentsTab
        agents={[agent]}
        currentAgent={null}
        onSelectAgent={onSelectAgent}
      />
    )
    fireEvent.click(screen.getByText('Helper'))
    expect(onSelectAgent).toHaveBeenCalledWith(agent)
  })

  it('highlights current agent', () => {
    render(
      <AgentsTab
        agents={[
          { id: 'a1', name: 'Helper', description: '', instructions: '' },
          { id: 'a2', name: 'Critic', description: '', instructions: '' },
        ]}
        currentAgent={{ id: 'a1', name: 'Helper', description: '', instructions: '' }}
        onSelectAgent={onSelectAgent}
      />
    )
    const buttons = screen.getAllByRole('button')
    const helperBtn = buttons.find(b => b.textContent?.includes('Helper'))
    expect(helperBtn?.className).toContain('bg-primary/10')
  })
})
