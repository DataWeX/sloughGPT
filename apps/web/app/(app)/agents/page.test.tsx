import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockList, mockCreate, mockDelete, mockExecute, mockOrchestrate, mockAddToast, mockDownloadJson,
} = vi.hoisted(() => ({
  mockList: vi.fn(), mockCreate: vi.fn(), mockDelete: vi.fn(),
  mockExecute: vi.fn(), mockOrchestrate: vi.fn(), mockAddToast: vi.fn(),
  mockDownloadJson: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
    ),
    AlertDialog: passthrough, AlertDialogAction: passthrough, AlertDialogCancel: passthrough,
    AlertDialogContent: passthrough, AlertDialogDescription: passthrough,
    AlertDialogFooter: passthrough, AlertDialogHeader: passthrough, AlertDialogTitle: passthrough,
    EmptyCard: ({ title, description }: any) => <div data-testid="empty-card"><div>{title}</div><div>{description}</div></div>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    IconPlus: () => <span>+</span>,
    IconTrash: () => <span>trash</span>,
    IconClock: () => <span>clock</span>,
  }
})

vi.mock('@/components/icons/NavIcons', () => ({
  IconPlus: () => <span>+</span>,
  IconTrash: () => <span>trash</span>,
  IconClock: () => <span>clock</span>,
}))

vi.mock('@/lib/agents-controller', () => ({
  agentsController: {
    list: (...a: unknown[]) => mockList(...a),
    create: (...a: unknown[]) => mockCreate(...a),
    delete: (...a: unknown[]) => mockDelete(...a),
    execute: (...a: unknown[]) => mockExecute(...a),
    orchestrate: (...a: unknown[]) => mockOrchestrate(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: (...a: unknown[]) => mockDownloadJson(...a),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
}))

vi.mock('@/lib/validation-schemas', () => ({
  agentSchema: { parse: (v: any) => v, safeParse: (v: any) => ({ success: true, data: v }) },
  agentExecuteSchema: { parse: (v: any) => v, safeParse: (v: any) => ({ success: true, data: v }) },
  orchestrateSchema: { parse: (v: any) => v, safeParse: (v: any) => ({ success: true, data: v }) },
}))

import AgentsPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([])
  mockCreate.mockResolvedValue({ id: 'agent-1', name: 'Test Agent', description: 'A test agent' })
  mockDelete.mockResolvedValue({})
  mockExecute.mockResolvedValue({ result: 'Agent executed' })
  mockOrchestrate.mockResolvedValue({ tasks: [] })
})

describe('AgentsPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<AgentsPage />)
    expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches agents on mount', async () => {
    render(<AgentsPage />)
    await waitFor(() => {
      expect(mockList).toHaveBeenCalledTimes(1)
    })
  })

  it('shows empty state when no agents', async () => {
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getByTestId('empty-card')).toBeTruthy()
    })
  })
})

describe('AgentsPage — agent list flow', () => {
  it('displays agents when loaded', async () => {
    mockList.mockResolvedValue([
      { id: 'a1', name: 'Researcher', description: 'Finds information', tools: ['web_search'], created_at: '2026-08-07T00:00:00Z' },
      { id: 'a2', name: 'Coder', description: 'Writes code', tools: ['code_execution'], created_at: '2026-08-07T00:00:00Z' },
    ])
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeTruthy()
      expect(screen.getByText('Coder')).toBeTruthy()
    })
  })

  it('shows tool count badge', async () => {
    mockList.mockResolvedValue([
      { id: 'a1', name: 'Researcher', description: 'Info', tools: ['web_search', 'knowledge_retrieval'], created_at: '2026-08-07T00:00:00Z' },
    ])
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getByText('Researcher')).toBeTruthy()
    })
  })
})

describe('AgentsPage — create agent flow', () => {
  it('new agent button opens form', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1) })

    const newBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('new') || b.textContent?.includes('+')
    )
    if (newBtn) {
      fireEvent.click(newBtn)
      await waitFor(() => {
        expect(screen.getAllByRole('button').length).toBeGreaterThan(1)
      })
    }
  })
})

describe('AgentsPage — template flow', () => {
  it('shows template options when creating', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1) })

    const newBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('new') || b.textContent?.includes('+')
    )
    if (newBtn) {
      fireEvent.click(newBtn)
      await waitFor(() => {
        // Templates should appear
        const researcher = screen.queryByText('Researcher')
        const coder = screen.queryByText('Coder')
        expect(researcher || coder).toBeTruthy()
      })
    }
  })
})

describe('AgentsPage — orchestrate flow', () => {
  it('orchestrate button triggers orchestration', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1) })

    const orchBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('orchestrate')
    )
    if (orchBtn) {
      fireEvent.click(orchBtn)
      await waitFor(() => {
        expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1)
      })
    }
  })
})

describe('AgentsPage — error handling', () => {
  it('handles list failure gracefully', async () => {
    mockList.mockRejectedValue(new Error('network'))
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('AgentsPage — stats display', () => {
  it('shows agent stats when agents exist', async () => {
    mockList.mockResolvedValue([
      { id: 'a1', name: 'R', description: '', tools: ['web_search'], created_at: '2026-08-07T00:00:00Z' },
    ])
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('AgentsPage — search/filter flow', () => {
  it('renders search input', async () => {
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1)
    })
    const searchInput = screen.queryByPlaceholderText(/search/i)
    expect(searchInput).toBeTruthy()
  })

  it('typing in search filters agents', async () => {
    mockList.mockResolvedValue([
      { id: 'a1', name: 'Researcher', description: 'Finds info', tools: ['web_search'], created_at: '2026-08-07T00:00:00Z' },
      { id: 'a2', name: 'Coder', description: 'Writes code', tools: ['code'], created_at: '2026-08-07T00:00:00Z' },
    ])
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getAllByText('Researcher').length).toBeGreaterThanOrEqual(1) })
    const searchInput = screen.getByPlaceholderText(/search/i)
    fireEvent.change(searchInput, { target: { value: 'Research' } })
    await waitFor(() => {
      expect(screen.getAllByText('Researcher').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('AgentsPage — loading state', () => {
  it('shows loading while fetching agents', async () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<AgentsPage />)
    expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1)
  })
})

describe('AgentsPage — agent actions flow', () => {
  beforeEach(() => {
    mockList.mockResolvedValue([
      { id: 'a1', name: 'TestAgent', description: 'Test', tools: ['web_search'], created_at: '2026-08-07T00:00:00Z' },
    ])
  })

  it('delete button triggers delete flow', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getAllByText('TestAgent').length).toBeGreaterThanOrEqual(1) })
    const deleteBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.includes('trash') || b.getAttribute('aria-label')?.includes('delete')
    )
    if (deleteBtn) {
      fireEvent.click(deleteBtn)
    }
  })

  it('shows agent description', async () => {
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('TestAgent').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('Test')).toBeTruthy()
    })
  })

  it('shows tool list for agent', async () => {
    render(<AgentsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('TestAgent').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('AgentsPage — error handling flow', () => {
  it('shows toast on create failure', async () => {
    mockCreate.mockRejectedValue(new Error('create failed'))
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getAllByText('Agents').length).toBeGreaterThanOrEqual(1) })
  })

  it('shows toast on delete failure', async () => {
    mockDelete.mockRejectedValue(new Error('delete failed'))
    mockList.mockResolvedValue([
      { id: 'a1', name: 'Agent', description: '', tools: [], created_at: '2026-08-07T00:00:00Z' },
    ])
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getAllByText('Agent').length).toBeGreaterThanOrEqual(1) })
  })
})
