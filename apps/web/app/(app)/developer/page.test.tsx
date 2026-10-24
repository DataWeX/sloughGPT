import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Captures the `r`-shortcut callback the page registers, so a test can fire it
// exactly the way a keypress would.
const { refreshState } = vi.hoisted(() => ({
  refreshState: { cb: null as (() => void) | null },
}))

vi.mock('@/lib/files-controller', () => ({
  filesController: { list: vi.fn() },
}))

vi.mock('@/lib/voice-controller', () => ({
  voiceController: { getStatus: vi.fn(), tts: vi.fn() },
}))

vi.mock('@/lib/http-client', () => ({
  // authFetch = fetch + URL resolution + auth headers. These tests drive
  // behavior through the global fetch stub, so the mock passes straight
  // through to it — a bare vi.fn() returns undefined and breaks the
  // transport-merged startup fetchers (they call authFetch, not fetch).
  authFetch: (url: string, init?: RequestInit) => globalThis.fetch(url, init),
}))

vi.mock('@/hooks/useRefreshShortcut', () => ({
  useRefreshShortcut: (cb: () => void) => {
    refreshState.cb = cb
  },
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
import { voiceController } from '@/lib/voice-controller'

describe('DeveloperPage', () => {
  beforeEach(() => {
    refreshState.cb = null
    vi.mocked(filesController.list).mockResolvedValue([])
    vi.mocked(voiceController.getStatus).mockResolvedValue({
      server_tts: false,
      model: null,
      error: null,
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
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

  it('refresh (`r`) re-reads voice status in place and never remounts the tab', async () => {
    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /voice/i }))

    // initial fetch lands
    await waitFor(() => expect(screen.getByText('TTS Unavailable')).toBeTruthy())

    // user-owned component state: a remount would wipe this
    const input = screen.getByLabelText(/text to speech input/i) as HTMLTextAreaElement
    await user.type(input, 'hello there')
    expect(input.value).toBe('hello there')

    // backend changes, then the user hits `r`
    vi.mocked(voiceController.getStatus).mockResolvedValue({
      server_tts: true,
      model: 'kokoro-en',
      error: null,
    })
    refreshState.cb?.()

    // content refreshed from the cache — no spinner swap, no remount
    await waitFor(() => expect(screen.getByText('TTS Available')).toBeTruthy())
    expect(screen.getByText('kokoro-en')).toBeTruthy()
    const after = screen.getByLabelText(/text to speech input/i) as HTMLTextAreaElement
    expect(after.value).toBe('hello there')
  })

  it('refresh (`r`) re-reads the startup tab in place without remounting it', async () => {
    const payload = (runs: number) => ({
      json: async () => ({
        data: {
          records: Array.from({ length: runs }, (_, i) => ({
            timestamp: 1700000000000 + i,
            total_duration: 10 + i,
            stage_durations: {},
            hook_durations: {},
            success: true,
          })),
        },
      }),
    })
    // history read #1 (mount) → 2 runs, every later read (the refresh) → 3 runs
    let historyReads = 0
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (String(url).includes('startup-history')) {
          historyReads += 1
          return Promise.resolve(payload(historyReads === 1 ? 2 : 3))
        }
        return Promise.resolve({ json: async () => ({ data: {} }) })
      }),
    )

    const user = userEvent.setup()
    render(<DeveloperPage />)
    await user.click(screen.getByRole('tab', { name: /startup/i }))

    await waitFor(() => expect(screen.getByText('2 runs')).toBeTruthy())

    // no `key=` remount mechanism: the same DOM node must survive a refresh
    const before = screen.getByText('2 runs').closest('div')!
    refreshState.cb?.()
    await waitFor(() => expect(screen.getByText('3 runs')).toBeTruthy())
    expect(screen.getByText('3 runs').closest('div')).toBe(before)
  })
})
