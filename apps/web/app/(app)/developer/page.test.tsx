import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('@/lib/files-controller', () => ({
  filesController: { list: vi.fn(), history: vi.fn() },
}))

vi.mock('@/lib/images-controller', () => ({
  imagesController: { gallery: vi.fn() },
}))

vi.mock('@/lib/voice-controller', () => ({
  voiceController: { getStatus: vi.fn(), tts: vi.fn() },
}))

vi.mock('@/lib/http-client', () => ({
  authFetch: vi.fn(),
}))

vi.mock('@/hooks/useRefreshShortcut', () => ({
  useRefreshShortcut: vi.fn(),
}))

vi.mock('@/components/shell/TerminalPanel', () => ({
  TerminalPanel: () => <div data-testid="terminal-panel" />,
}))

vi.mock('@/components/files/FileStatsCard', () => ({
  FileStatsCard: () => <div data-testid="file-stats-card" />,
}))

vi.mock('@/lib/cache/api-hooks', () => ({
  useModels: () => ({ data: [], isLoading: false, refetch: vi.fn() }),
}))

vi.mock('@/components/models/ModelCatalogCard', () => ({
  default: () => <div data-testid="model-catalog-card" />,
}))

vi.mock('@/components/models/QuantizationCard', () => ({
  default: () => <div data-testid="quantization-card" />,
}))

vi.mock('@/components/models/DownloadsCard', () => ({
  default: () => <div data-testid="downloads-card" />,
}))

vi.mock('@/components/models/EngineStatusCard', () => ({
  default: () => <div data-testid="engine-status-card" />,
}))

import DeveloperPage from './page'
import { filesController } from '@/lib/files-controller'
import { imagesController } from '@/lib/images-controller'
import { voiceController } from '@/lib/voice-controller'
import { PUBLIC_API_URL } from '@/lib/config'

describe('DeveloperPage', () => {
  beforeEach(() => {
    vi.mocked(filesController.list).mockResolvedValue([])
    vi.mocked(voiceController.getStatus).mockResolvedValue({
      server_tts: false,
      model: null,
      error: null,
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', () => {
    render(<DeveloperPage />)
    expect(screen.getAllByText('Developer').length).toBeGreaterThanOrEqual(1)
  })

  it('renders tab list with four tabs', () => {
    render(<DeveloperPage />)
    expect(screen.getByRole('tab', { name: /terminal/i })).toBeTruthy()
    expect(screen.getByRole('tab', { name: /files/i })).toBeTruthy()
    expect(screen.getByRole('tab', { name: /voice/i })).toBeTruthy()
    expect(screen.getByRole('tab', { name: /api/i })).toBeTruthy()
  })

  it('defaults to shell tab', () => {
    render(<DeveloperPage />)
    expect(screen.getByTestId('terminal-panel')).toBeTruthy()
  })

  it('switches to files tab', async () => {
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /files/i }))
    await waitFor(() => {
      expect(screen.getByTestId('file-stats-card')).toBeTruthy()
    })
    expect(screen.getByText('Documents')).toBeTruthy()
  })

  it('switches to API tab', async () => {
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /api/i }))
    expect(screen.getByText('API Playground')).toBeTruthy()
    expect(screen.getByLabelText(/http method/i)).toBeTruthy()
    expect(screen.getByLabelText(/request path/i)).toBeTruthy()
  })

  it('shows request body textarea for POST method', async () => {
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /api/i }))
    const methodSelect = screen.getByLabelText(/http method/i)
    await user.selectOptions(methodSelect, 'POST')
    expect(screen.getByLabelText(/request body/i)).toBeTruthy()
  })

  it('hides request body for GET method', async () => {
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /api/i }))
    expect(screen.queryByLabelText(/request body/i)).toBeNull()
  })

  it('shows authorization header input', async () => {
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /api/i }))
    expect(screen.getByLabelText(/authorization header/i)).toBeTruthy()
  })

  it('renders Models tab with catalog cards', async () => {
    const user = userEvent.setup()
    render(<DeveloperPage />)
    expect(screen.getByRole('tab', { name: /models/i })).toBeTruthy()
    await user.click(screen.getByRole('tab', { name: /models/i }))
    await waitFor(() => {
      expect(screen.getByTestId('model-catalog-card')).toBeTruthy()
    })
    expect(screen.getByTestId('quantization-card')).toBeTruthy()
    expect(screen.getByTestId('downloads-card')).toBeTruthy()
    expect(screen.getByTestId('engine-status-card')).toBeTruthy()
  })
})

describe('DeveloperPage — asset store tabs (Images | History)', () => {
  const mockGallery = vi.mocked(imagesController.gallery)
  const mockHistory = vi.mocked(filesController.history)

  beforeEach(() => {
    vi.mocked(filesController.list).mockResolvedValue([])
    vi.mocked(voiceController.getStatus).mockResolvedValue({
      server_tts: false,
      model: null,
      error: null,
    })
    mockGallery.mockResolvedValue({ images: [] })
    mockHistory.mockResolvedValue([])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('exposes Images and History tabs in the live tab strip', () => {
    render(<DeveloperPage />)
    expect(screen.getByRole('tab', { name: /images/i })).toBeTruthy()
    expect(screen.getByRole('tab', { name: /history/i })).toBeTruthy()
  })

  it('Images tab fetches the gallery and renders API-origin img src', async () => {
    mockGallery.mockResolvedValue({
      images: [{ id: 'generated_live99', path: '/images/gallery/generated_live99.png', created: 1761000000 }],
    })
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /images/i }))
    await waitFor(() => expect(mockGallery).toHaveBeenCalledTimes(1))
    const img = await screen.findByAltText('generated_live99')
    expect(img.getAttribute('src')).toBe(`${PUBLIC_API_URL}/images/gallery/generated_live99.png`)
  })

  it('History tab fetches and renders asset events', async () => {
    mockHistory.mockResolvedValue([
      { ts: 1761000000, action: 'put', original_name: 'report.pdf', origin: 'upload' },
      { ts: 1760000000, action: 'delete', original_name: 'old.txt', origin: 'upload' },
    ])
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /history/i }))
    await waitFor(() => expect(mockHistory).toHaveBeenCalledTimes(1))
    expect(await screen.findByText('report.pdf')).toBeTruthy()
    expect(screen.getByText('old.txt')).toBeTruthy()
    expect(screen.getByTestId('history-stream')).toBeTruthy()
  })
})
