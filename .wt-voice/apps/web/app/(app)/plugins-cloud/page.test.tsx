import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ title, children }: any) => (
    <div>
      <div>{title}</div>
      {children}
    </div>
  ),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
    ),
    Badge: ({ children, variant }: any) => <span data-variant={variant}>{children}</span>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

import PluginsCloudPage from './page'

describe('PluginsCloudPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockImplementation(async (url: string) => {
      if (url === '/cloud-training/jobs') return { jobs: [] }
      if (url === '/plugins') return { plugins: [] }
      return {}
    })
  })

  it('renders without crashing', async () => {
    render(<PluginsCloudPage />)
    expect(screen.getAllByText('Plugins & Cloud Training').length).toBeGreaterThanOrEqual(1)
  })

  it('renders page title', () => {
    render(<PluginsCloudPage />)
    expect(screen.getByText('Plugins & Cloud Training')).toBeTruthy()
  })

  it('shows loading state initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<PluginsCloudPage />)
    expect(screen.getByTestId('skeleton')).toBeTruthy()
  })

  it('shows cloud tab content by default', async () => {
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
  })

  it('switches to plugins tab', async () => {
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
    const pluginsTab = screen.getByText('Plugins')
    await user.click(pluginsTab)
    expect(screen.getByText('Installed Plugins')).toBeTruthy()
  })

  it('shows empty state for cloud jobs when no data', async () => {
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('No training jobs yet.')).toBeTruthy()
    })
  })

  it('shows empty state for plugins when no data', async () => {
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
    await user.click(screen.getByText('Plugins'))
    expect(screen.getByText(/No plugins installed/)).toBeTruthy()
  })

  it('displays cloud jobs after data loads', async () => {
    mockApiGet.mockImplementation(async (url: string) => {
      if (url === '/cloud-training/jobs') return {
        jobs: [{ job_id: 'job-1', provider: 'aws', status: 'completed', progress: 100 }],
      }
      return { plugins: [] }
    })
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('job-1')).toBeTruthy()
    })
    expect(screen.getByText('completed')).toBeTruthy()
  })

  it('displays plugins after data loads', async () => {
    mockApiGet.mockImplementation(async (url: string) => {
      if (url === '/plugins') return {
        plugins: [{ name: 'my-plugin', version: '1.0.0', description: 'desc', author: 'test', enabled: true }],
      }
      return { jobs: [] }
    })
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
    await user.click(screen.getByText('Plugins'))
    expect(screen.getByText('my-plugin')).toBeTruthy()
    expect(screen.getByText('v1.0.0')).toBeTruthy()
  })

  it('shows success toast when job is submitted', async () => {
    mockApiPost.mockResolvedValue({})
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
    const input = screen.getByPlaceholderText('e.g. my-dataset')
    await user.type(input, 'my-dataset')
    await user.click(screen.getByText('Submit Job'))
    expect(mockApiPost).toHaveBeenCalledWith('/cloud-training/submit', { provider: 'local', dataset_id: 'my-dataset' })
    expect(mockAddToast).toHaveBeenCalledWith('Job submitted', 'success')
  })

  it('shows error toast when job submission fails', async () => {
    mockApiPost.mockRejectedValue(new Error('network error'))
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
    const input = screen.getByPlaceholderText('e.g. my-dataset')
    await user.type(input, 'my-dataset')
    await user.click(screen.getByText('Submit Job'))
    expect(mockAddToast).toHaveBeenCalledWith('Failed to submit', 'error')
  })

  it('disables submit button when dataset ID is empty', async () => {
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
    const button = screen.getByText('Submit Job')
    expect(button).toBeDisabled()
  })

  it('enables submit button when dataset ID is provided', async () => {
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('Submit Training Job')).toBeTruthy()
    })
    const input = screen.getByPlaceholderText('e.g. my-dataset')
    await user.type(input, 'ds-123')
    const button = screen.getByText('Submit Job')
    expect(button).not.toBeDisabled()
  })

  it('fetches data on mount', async () => {
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/cloud-training/jobs')
      expect(mockApiGet).toHaveBeenCalledWith('/plugins')
    })
  })
})
