import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockGetInspector = vi.fn()
const mockRegenerate = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/session-controller', () => ({
  sessionController: {
    getInspector: (...args: unknown[]) => mockGetInspector(...args),
    regenerate: (...args: unknown[]) => mockRegenerate(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => mockAddToast,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant, className }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, placeholder, className, onKeyDown }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} onKeyDown={onKeyDown} />
    ),
    Label: ({ children, className }: any) => <label className={className}>{children}</label>,
    Textarea: ({ value, onChange, rows, className }: any) => (
      <textarea value={value} onChange={onChange} rows={rows} className={className} />
    ),
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import SessionPage from './page'

const mockInspector = {
  session: { id: 's1', message_count: 5, messages: [{ role: 'user', content: 'Hello' }, { role: 'assistant', content: 'Hi there' }] },
  knowledge: { total_facts: 3, topics: ['general', 'coding'] },
  traits: { creativity: 0.8 },
  modes: { tone: 'friendly' },
  feedback: { total: 10, thumbs_up: 7, thumbs_down: 3 },
  workspace: { working_memory: ['topic: AI'], semantic_keys: ['user_pref'], episodic_count: 12, sensory_buffer_size: 5, system_prompt: 'You are helpful.' },
  elapsed_ms: 42,
}

describe('SessionPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders title and subtitle', async () => {
    render(<SessionPage />)
    expect(screen.getByText('Session Inspector')).toBeInTheDocument()
    expect(screen.getByText('Debug and inspect chat session state')).toBeInTheDocument()
  })

  it('has inspect button disabled when no session ID', async () => {
    render(<SessionPage />)
    expect(screen.getByText('Inspect')).toBeDisabled()
  })

  it('calls getInspector on Inspect click', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(mockGetInspector).toHaveBeenCalledWith('s1')
    }, { timeout: 5000 })
  })

  it('displays inspector stats', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(screen.getByText('5')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('10')).toBeInTheDocument()
    expect(screen.getByText('42ms')).toBeInTheDocument()
  })

  it('displays workspace data', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(screen.getByText('Episodic Memory')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.getByText('Sensory Buffer')).toBeInTheDocument()
    expect(screen.getByText('topic: AI')).toBeInTheDocument()
    expect(screen.getByText('user_pref')).toBeInTheDocument()
  })

  it('displays modes and traits', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(screen.getByText('tone')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('friendly')).toBeInTheDocument()
  })

  it('displays knowledge topics', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(screen.getByText('general')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('coding')).toBeInTheDocument()
  })

  it('displays feedback stats', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(screen.getByText('7')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('70.0%')).toBeInTheDocument()
  })

  it('displays messages', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(screen.getByText('Recent Messages (2)')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('Hello')).toBeInTheDocument()
    expect(screen.getByText('Hi there')).toBeInTheDocument()
  })

  it('shows error toast on inspect failure', async () => {
    mockGetInspector.mockRejectedValue(new Error('not found'))
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 'bad' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not load session: not found', 'error')
    }, { timeout: 5000 })
  })

  it('calls regenerate on Regenerate click', async () => {
    mockRegenerate.mockResolvedValue({ status: 'ok' })
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Regenerate Last Response'))
    await waitFor(() => {
      expect(mockRegenerate).toHaveBeenCalledWith('s1')
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Regeneration started', 'success')
  })

  it('shows error on regenerate failure', async () => {
    mockRegenerate.mockRejectedValue(new Error('timeout'))
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Regenerate Last Response'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Regeneration failed: timeout', 'error')
    }, { timeout: 5000 })
  })

  it('refreshes inspector on Refresh click', async () => {
    mockGetInspector.mockResolvedValue(mockInspector)
    render(<SessionPage />)
    fireEvent.change(screen.getByPlaceholderText('Enter session ID...'), { target: { value: 's1' } })
    fireEvent.click(screen.getByText('Inspect'))
    await waitFor(() => {
      expect(mockGetInspector).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockGetInspector).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })
})
