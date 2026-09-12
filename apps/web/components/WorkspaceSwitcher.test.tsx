import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconChevronDown: (props: any) => <svg data-testid="icon-chevron" {...props} />,
  IconBuilding: (props: any) => <svg data-testid="icon-building" {...props} />,
}))

const mockSwitchWorkspace = vi.fn()
let mockWorkspaces = [
  { id: 'ws-1', name: 'Alpha', description: 'First workspace' },
  { id: 'ws-2', name: 'Beta', description: 'Second workspace' },
]
let mockCurrentWorkspace: { id: string; name: string; description: string } | null = {
  id: 'ws-1', name: 'Alpha', description: 'First workspace',
}

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: any) => any) => {
    const state = {
      workspaces: mockWorkspaces,
      currentWorkspace: mockCurrentWorkspace,
      switchWorkspace: mockSwitchWorkspace,
    }
    return selector ? selector(state) : state
  },
}))

import { WorkspaceSwitcher } from './WorkspaceSwitcher'

afterEach(() => {
  cleanup()
  mockWorkspaces = [
    { id: 'ws-1', name: 'Alpha', description: 'First workspace' },
    { id: 'ws-2', name: 'Beta', description: 'Second workspace' },
  ]
  mockCurrentWorkspace = { id: 'ws-1', name: 'Alpha', description: 'First workspace' }
  vi.clearAllMocks()
})

describe('WorkspaceSwitcher', () => {
  it('renders the current workspace name in the trigger button', () => {
    render(<WorkspaceSwitcher />)
    const trigger = screen.getAllByText('Alpha')[0].closest('button')
    expect(trigger).toBeInTheDocument()
  })

  it('renders all workspaces in the dropdown', () => {
    render(<WorkspaceSwitcher />)
    expect(screen.getAllByText('Alpha').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Beta')).toBeInTheDocument()
  })

  it('calls switchWorkspace when a workspace option is clicked', () => {
    render(<WorkspaceSwitcher />)
    fireEvent.click(screen.getByText('Beta'))
    expect(mockSwitchWorkspace).toHaveBeenCalledWith('ws-2')
  })

  it('returns null when there is only one workspace', () => {
    mockWorkspaces = [{ id: 'ws-1', name: 'Solo', description: '' }]
    mockCurrentWorkspace = { id: 'ws-1', name: 'Solo', description: '' }
    const { container } = render(<WorkspaceSwitcher />)
    expect(container.innerHTML).toBe('')
  })
})
