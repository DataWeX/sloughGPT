import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/controllers', () => ({
  agentsController: {
    list: vi.fn(),
  },
  modelController: {
    list: vi.fn(),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

import { MentionMenu } from './MentionMenu'
import { agentsController, modelController } from '@/lib/controllers'

const mockAgents = [
  { id: 'a1', name: 'Researcher', description: 'Finds information' },
  { id: 'a2', name: 'Writer', description: 'Writes content' },
]
const mockModels = [
  { id: 'm1', name: 'GPT-2', description: 'Small model' },
  { id: 'm2', name: 'Qwen', description: 'Chat model' },
]

beforeEach(() => {
  vi.mocked(agentsController.list).mockResolvedValue(mockAgents as any)
  vi.mocked(modelController.list).mockResolvedValue(mockModels as any)
})

afterEach(() => cleanup())

function renderMenu(value = '@', props = {}) {
  const onInsert = vi.fn()
  const onClose = vi.fn()
  const result = render(
    <MentionMenu value={value} onInsert={onInsert} onClose={onClose} {...props} />
  )
  return { ...result, onInsert, onClose }
}

describe('MentionMenu', () => {
  it('returns null while loading', () => {
    const { container } = renderMenu()
    expect(container.innerHTML).toBe('')
  })

  it('renders items after loading', async () => {
    renderMenu()
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeDefined()
    })
    expect(screen.getByText('Writer')).toBeDefined()
    expect(screen.getByText('GPT-2')).toBeDefined()
    expect(screen.getByText('Qwen')).toBeDefined()
  })

  it('shows agent/model type badges', async () => {
    renderMenu()
    await waitFor(() => {
      expect(screen.getAllByText('Agent').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Model').length).toBeGreaterThanOrEqual(1)
  })

  it('shows description when available', async () => {
    renderMenu()
    await waitFor(() => {
      expect(screen.getByText('Finds information')).toBeDefined()
    })
  })

  it('filters items by fuzzy query', async () => {
    renderMenu('@wri')
    await waitFor(() => {
      expect(screen.getByText('Writer')).toBeDefined()
    })
    expect(screen.queryByText('Researcher')).toBeNull()
  })

  it('returns null when query matches nothing', async () => {
    const { container } = renderMenu('@zzzzz')
    await waitFor(() => {
      // After loading, check if menu is hidden (no matching items)
      const listbox = container.querySelector('[role="listbox"]')
      // zzzzz doesn't match any agent/model name or description, so menu should be empty
      expect(listbox).toBeNull()
    })
  })

  it('calls onInsert with @name when item is clicked', async () => {
    const { onInsert, onClose } = renderMenu()
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeDefined()
    })
    fireEvent.click(screen.getByText('Researcher'))
    expect(onInsert).toHaveBeenCalledWith('@Researcher ')
    expect(onClose).toHaveBeenCalled()
  })

  it('navigates with ArrowDown and ArrowUp', async () => {
    renderMenu()
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeDefined()
    })
    const listbox = screen.getByRole('listbox')
    fireEvent.keyDown(listbox, { key: 'ArrowDown' })
    // selectedIndex moves to 1
    const options = screen.getAllByRole('option')
    expect(options[1].getAttribute('aria-selected')).toBe('true')
    fireEvent.keyDown(listbox, { key: 'ArrowUp' })
    expect(options[0].getAttribute('aria-selected')).toBe('true')
  })

  it('selects item on Enter key', async () => {
    const { onInsert } = renderMenu()
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeDefined()
    })
    const listbox = screen.getByRole('listbox')
    fireEvent.keyDown(listbox, { key: 'Enter' })
    expect(onInsert).toHaveBeenCalledWith('@Researcher ')
  })

  it('selects item on Tab key', async () => {
    const { onInsert } = renderMenu()
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeDefined()
    })
    const listbox = screen.getByRole('listbox')
    fireEvent.keyDown(listbox, { key: 'Tab' })
    expect(onInsert).toHaveBeenCalled()
  })

  it('calls onClose on Escape', async () => {
    const { onClose } = renderMenu()
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeDefined()
    })
    const listbox = screen.getByRole('listbox')
    fireEvent.keyDown(listbox, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('shows item count like 1/4', async () => {
    renderMenu()
    await waitFor(() => {
      expect(screen.getByText('1/4')).toBeDefined()
    })
  })

  it('handles agent list returning non-array gracefully', async () => {
    vi.mocked(agentsController.list).mockResolvedValue({ error: 'fail' } as any)
    renderMenu()
    await waitFor(() => {
      expect(screen.getByText('GPT-2')).toBeDefined()
    })
    // Models still show, agents are empty
    expect(screen.queryByText('Researcher')).toBeNull()
  })
})
