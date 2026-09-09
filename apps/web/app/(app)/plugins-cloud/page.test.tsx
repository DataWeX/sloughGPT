import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockListCloudJobs = vi.fn()
const mockListPlugins = vi.fn()

vi.mock('@/lib/plugins-cloud-controller', () => ({
  pluginsCloudController: {
    listCloudJobs: (...args: unknown[]) => mockListCloudJobs(...args),
    listPlugins: (...args: unknown[]) => mockListPlugins(...args),
    submitCloudJob: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

import PluginsCloudPage from './page'

describe('PluginsCloudPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockListCloudJobs.mockResolvedValue([])
    mockListPlugins.mockResolvedValue([])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<PluginsCloudPage />)
    await screen.findAllByText('Plugins & Cloud Training')
  })

  it('renders cloud and plugins tabs', async () => {
    render(<PluginsCloudPage />)
    await screen.findAllByText('Plugins & Cloud Training')
    expect(screen.getByText('Cloud Training')).toBeTruthy()
    expect(screen.getByText('Plugins')).toBeTruthy()
  })

  it('defaults to cloud tab', async () => {
    render(<PluginsCloudPage />)
    await screen.findAllByText('Plugins & Cloud Training')
    expect(screen.getByText('Submit Training Job')).toBeTruthy()
  })

  it('shows submit form on cloud tab', async () => {
    render(<PluginsCloudPage />)
    await screen.findAllByText('Plugins & Cloud Training')
    expect(screen.getByText('Provider')).toBeTruthy()
    expect(screen.getByText('Dataset ID')).toBeTruthy()
    expect(screen.getByText('Submit Job')).toBeTruthy()
  })

  it('fetches data on mount', async () => {
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(mockListCloudJobs).toHaveBeenCalled()
      expect(mockListPlugins).toHaveBeenCalled()
    })
  })

  it('shows empty state when no training jobs', async () => {
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('No training jobs yet.')).toBeTruthy()
    })
  })

  it('switches to plugins tab', async () => {
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await screen.findAllByText('Plugins & Cloud Training')
    await user.click(screen.getByText('Plugins'))
    expect(screen.getByText('Installed Plugins')).toBeTruthy()
  })

  it('shows empty state when no plugins', async () => {
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await screen.findAllByText('Plugins & Cloud Training')
    await user.click(screen.getByText('Plugins'))
    await waitFor(() => {
      expect(screen.getByText(/no plugins installed/i)).toBeTruthy()
    })
  })

  it('displays cloud jobs when present', async () => {
    mockListCloudJobs.mockResolvedValue([
      { job_id: 'job-1', provider: 'aws', status: 'running', dataset_id: 'ds-1' },
    ])
    render(<PluginsCloudPage />)
    await waitFor(() => {
      expect(screen.getByText('job-1')).toBeTruthy()
      expect(screen.getByText('running')).toBeTruthy()
    })
  })

  it('displays plugins when present', async () => {
    mockListPlugins.mockResolvedValue([
      { name: 'my-plugin', version: '1.0.0', author: 'test', enabled: true },
    ])
    const user = userEvent.setup()
    render(<PluginsCloudPage />)
    await screen.findAllByText('Plugins & Cloud Training')
    await user.click(screen.getByText('Plugins'))
    await waitFor(() => {
      expect(screen.getByText('my-plugin')).toBeTruthy()
      expect(screen.getByText('v1.0.0')).toBeTruthy()
    })
  })
})
