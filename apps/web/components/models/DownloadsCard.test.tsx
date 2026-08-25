// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import DownloadsCard from './DownloadsCard'

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    listDownloads: vi.fn(),
    cancelDownload: vi.fn(),
    retryDownload: vi.fn(),
    verifyDownload: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

import { modelController } from '@/lib/model-controller'

describe('DownloadsCard', () => {
  afterEach(cleanup)

  it('renders nothing when no downloads', async () => {
    vi.mocked(modelController.listDownloads).mockResolvedValue({ downloads: [], count: 0 })
    const { container } = render(<DownloadsCard />)
    await vi.waitFor(() => {
      expect(container.innerHTML).toBe('')
    })
  })

  it('renders active download with progress', async () => {
    vi.mocked(modelController.listDownloads).mockResolvedValue({
      downloads: [{ model_id: 'gpt2', status: 'downloading', progress: 0.5, bytes_downloaded: 500, total_bytes: 1000, speed_bps: 200 }],
      count: 1,
    })
    render(<DownloadsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('gpt2')).toBeDefined()
      expect(screen.getByText('downloading')).toBeDefined()
    })
  })

  it('renders failed download with retry/verify buttons', async () => {
    vi.mocked(modelController.listDownloads).mockResolvedValue({
      downloads: [{ model_id: 'bert', status: 'failed', progress: 0, bytes_downloaded: 100, total_bytes: 500, speed_bps: 0 }],
      count: 1,
    })
    render(<DownloadsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('bert')).toBeDefined()
      expect(screen.getByText('failed')).toBeDefined()
      expect(screen.getByText('Retry')).toBeDefined()
      expect(screen.getByText('Verify')).toBeDefined()
    })
  })

  it('renders cancel button for active downloads', async () => {
    vi.mocked(modelController.listDownloads).mockResolvedValue({
      downloads: [{ model_id: 'llama', status: 'downloading', progress: 0.3, bytes_downloaded: 300, total_bytes: 1000, speed_bps: 100 }],
      count: 1,
    })
    render(<DownloadsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('Cancel')).toBeDefined()
    })
  })
})
