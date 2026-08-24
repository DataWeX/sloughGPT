import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
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
    Input: ({ value, onChange, placeholder, className, id, type, min, max, step, disabled }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} id={id}
        type={type} min={min} max={max} step={step} disabled={disabled} />
    ),
    Label: ({ children, htmlFor, variant }: any) => <label htmlFor={htmlFor} data-variant={variant}>{children}</label>,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import SelfTrainPage from './page'

describe('SelfTrainPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders title and subtitle', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    render(<SelfTrainPage />)
    expect(screen.getByText('Self-train')).toBeInTheDocument()
    expect(screen.getByText('Autonomous self-training subprocess')).toBeInTheDocument()
  })

  it('fetches status on mount', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/self-train/status')
    }, { timeout: 5000 })
  })

  it('shows not started status', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Not started')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('shows running status', async () => {
    mockApiGet.mockResolvedValue({ status: 'running', pid: 12345, history: ['line 1', 'line 2'] })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Running')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('12345')).toBeInTheDocument()
  })

  it('shows exited status', async () => {
    mockApiGet.mockResolvedValue({ status: 'exited', returncode: 0, history: ['done'] })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Exited')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('shows training history', async () => {
    mockApiGet.mockResolvedValue({ status: 'running', history: ['Epoch 1/10', 'Epoch 2/10'] })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Epoch 1/10')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('Epoch 2/10')).toBeInTheDocument()
  })

  it('shows no history message', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('No history yet.')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('starts self-training', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    mockApiPost.mockResolvedValue({})
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Not started')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Start self-training'))
    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/self-train/start', expect.anything())
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Self-training started', 'success')
  })

  it('stops self-training when running', async () => {
    mockApiGet.mockResolvedValue({ status: 'running', pid: 123 })
    mockApiPost.mockResolvedValue({})
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Running')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Stop'))
    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/self-train/stop')
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Self-training stopped', 'success')
  })

  it('sends model and temperature in start body', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    mockApiPost.mockResolvedValue({})
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Not started')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.change(screen.getByLabelText('Model (optional)'), { target: { value: 'gpt2' } })
    fireEvent.change(screen.getByLabelText('Temperature'), { target: { value: '0.9' } })
    fireEvent.click(screen.getByText('Start self-training'))
    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/self-train/start', expect.objectContaining({ model: 'gpt2', temperature: 0.9 }))
    }, { timeout: 5000 })
  })

  it('toggles forever mode', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(screen.getByText('Single pass')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Single pass'))
    expect(screen.getByText('Train forever')).toBeInTheDocument()
  })

  it('shows error on status fetch failure', async () => {
    mockApiGet.mockRejectedValue(new Error('network'))
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not check status', 'error')
    }, { timeout: 5000 })
  })

  it('refreshes on Refresh click', async () => {
    mockApiGet.mockResolvedValue({ status: 'not_started' })
    render(<SelfTrainPage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })
})
