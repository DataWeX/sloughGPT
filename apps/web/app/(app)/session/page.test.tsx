import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockGetInspector = vi.fn()
const mockRegenerate = vi.fn()

vi.mock('@/lib/session-controller', () => ({
  sessionController: {
    getInspector: (...args: unknown[]) => mockGetInspector(...args),
    regenerate: (...args: unknown[]) => mockRegenerate(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant }: any) => <button onClick={onClick} disabled={disabled} data-variant={variant}>{children}</button>,
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    Input: ({ value, onChange, placeholder, onKeyDown, type }: any) => <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} type={type} />,
    Label: ({ children }: any) => <label>{children}</label>,
    Textarea: ({ value, onChange, placeholder, rows }: any) => <textarea value={value} onChange={onChange} placeholder={placeholder} rows={rows} />,
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ children }: any) => <span>{children}</span>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh" />,
    IconTrash: () => <span data-testid="icon-trash" />,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialogCancel: ({ children }: any) => <button>{children}</button>,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <p>{children}</p>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p>{headerRight && <div>{headerRight}</div>}{children}</div>
  ),
}))

import SessionPage from './page'

describe('SessionPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders page title and subtitle', async () => {
    render(<SessionPage />)
    expect(screen.getByText('Session Inspector')).toBeInTheDocument()
    expect(screen.getByText('Debug and inspect chat session state')).toBeInTheDocument()
  })

  it('inspects session', async () => {
    mockGetInspector.mockResolvedValue({
      session: { message_count: 5, messages: [{ role: 'user', content: 'Hello' }, { role: 'assistant', content: 'Hi' }] },
      knowledge: { total_facts: 3, topics: ['general'] },
      feedback: { total: 10, thumbs_up: 7, thumbs_down: 3 },
      workspace: { episodic_count: 2, sensory_buffer_size: 1, working_memory: ['task'], semantic_keys: ['key1'], system_prompt: 'You are helpful.' },
      modes: { style: 'formal' },
      traits: { curiosity: 0.8 },
      elapsed_ms: 42,
    })
    render(<SessionPage />)

    const input = screen.getByPlaceholderText('Enter session ID...')
    fireEvent.change(input, { target: { value: 'sess-123' } })
    fireEvent.click(screen.getByText('Inspect'))

    await waitFor(() => {
      expect(mockGetInspector).toHaveBeenCalledWith('sess-123')
      expect(screen.getByText('5')).toBeInTheDocument()
      expect(screen.getByText('10')).toBeInTheDocument()
      expect(screen.getByText('42ms')).toBeInTheDocument()
    })
  })

  it('shows inspector data', async () => {
    mockGetInspector.mockResolvedValue({
      session: { message_count: 2, messages: [{ role: 'user', content: 'Test' }] },
      knowledge: { total_facts: 1, topics: ['code'] },
      feedback: { total: 5, thumbs_up: 3, thumbs_down: 2 },
      workspace: { episodic_count: 1, sensory_buffer_size: 0, working_memory: [], semantic_keys: [], system_prompt: '' },
      modes: {},
      traits: {},
      elapsed_ms: 10,
    })
    render(<SessionPage />)

    const input = screen.getByPlaceholderText('Enter session ID...')
    fireEvent.change(input, { target: { value: 'sess-456' } })
    fireEvent.click(screen.getByText('Inspect'))

    await waitFor(() => {
      expect(screen.getByText('Workspace')).toBeInTheDocument()
      expect(screen.getByText('Knowledge')).toBeInTheDocument()
      expect(screen.getByText('Feedback')).toBeInTheDocument()
      expect(screen.getByText('Modes & Traits')).toBeInTheDocument()
    })
  })

  it('regenerate', async () => {
    mockGetInspector.mockResolvedValue({
      session: { message_count: 1, messages: [] },
      knowledge: { total_facts: 0, topics: [] },
      feedback: { total: 0, thumbs_up: 0, thumbs_down: 0 },
      workspace: { episodic_count: 0, sensory_buffer_size: 0, working_memory: [], semantic_keys: [], system_prompt: '' },
      modes: {},
      traits: {},
      elapsed_ms: 5,
    })
    mockRegenerate.mockResolvedValue({ status: 'ok' })
    render(<SessionPage />)

    const input = screen.getByPlaceholderText('Enter session ID...')
    fireEvent.change(input, { target: { value: 'sess-789' } })
    fireEvent.click(screen.getByText('Regenerate Last Response'))

    await waitFor(() => {
      expect(mockRegenerate).toHaveBeenCalledWith('sess-789')
    })
  })
})