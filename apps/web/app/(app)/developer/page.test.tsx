import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('@/lib/files-controller', () => ({
  filesController: { list: vi.fn() },
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

import DeveloperPage from './page'
import { filesController } from '@/lib/files-controller'
import { voiceController } from '@/lib/voice-controller'

describe('DeveloperPage', () => {
  beforeEach(() => {
    vi.mocked(filesController.list).mockResolvedValue([])
    vi.mocked(voiceController.getStatus).mockResolvedValue({ server_tts: false, model: null, error: null })
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
})
