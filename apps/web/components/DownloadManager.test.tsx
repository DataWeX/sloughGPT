// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockGetDownloadStatus, mockCancelDownload } = vi.hoisted(() => ({
  mockGetDownloadStatus: vi.fn(),
  mockCancelDownload: vi.fn(),
}))

vi.mock('@/lib/download-controller', () => ({
  getDownloadStatus: mockGetDownloadStatus,
  cancelDownload: mockCancelDownload,
}))

import { DownloadManager } from './DownloadManager'

const baseProgress = {
  status: 'downloading' as const,
  percentage: 45,
  bytes_downloaded: 500_000_000,
  total_bytes: 1_000_000_000,
  speed_mb_per_sec: 12.5,
  eta_seconds: 40,
  current_file: 'model.safetensors',
}

describe('DownloadManager', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('shows download prompt when no progress', () => {
    mockGetDownloadStatus.mockResolvedValue(null)
    render(<DownloadManager modelId="gpt2" />)
    expect(screen.getByText('Download model')).toBeDefined()
  })

  it('shows size when provided', () => {
    mockGetDownloadStatus.mockResolvedValue(null)
    render(<DownloadManager modelId="gpt2" sizeGb={1.5} />)
    expect(screen.getByText('1.50 GB')).toBeDefined()
  })

  it('starts polling on download click', async () => {
    mockGetDownloadStatus.mockResolvedValue(baseProgress)
    render(<DownloadManager modelId="gpt2" />)
    fireEvent.click(screen.getByText('Download model'))
    expect(await screen.findByText('Downloading')).toBeDefined()
  })

  it('shows progress percentage', async () => {
    mockGetDownloadStatus.mockResolvedValue(baseProgress)
    render(<DownloadManager modelId="gpt2" />)
    fireEvent.click(screen.getByText('Download model'))
    expect(await screen.findByText('45.0%')).toBeDefined()
  })

  it('shows speed and ETA', async () => {
    mockGetDownloadStatus.mockResolvedValue(baseProgress)
    render(<DownloadManager modelId="gpt2" />)
    fireEvent.click(screen.getByText('Download model'))
    expect(await screen.findByText(/12.5 MB\/s/)).toBeDefined()
    expect(await screen.findByText(/40s left/)).toBeDefined()
  })

  it('shows ready state on complete', async () => {
    mockGetDownloadStatus.mockResolvedValue({ ...baseProgress, status: 'complete', percentage: 100 })
    render(<DownloadManager modelId="gpt2" />)
    fireEvent.click(screen.getByText('Download model'))
    expect(await screen.findByText('Ready')).toBeDefined()
    expect(await screen.findByText('Model downloaded and loaded successfully.')).toBeDefined()
  })

  it('shows cancelled state', async () => {
    mockGetDownloadStatus.mockResolvedValue({ ...baseProgress, status: 'cancelled' })
    render(<DownloadManager modelId="gpt2" />)
    fireEvent.click(screen.getByText('Download model'))
    expect(await screen.findByText('Cancelled')).toBeDefined()
  })

  it('shows failed state with error', async () => {
    mockGetDownloadStatus.mockResolvedValue({ ...baseProgress, status: 'failed', error: 'Connection lost' })
    render(<DownloadManager modelId="gpt2" />)
    fireEvent.click(screen.getByText('Download model'))
    expect(await screen.findByText('Failed')).toBeDefined()
    expect(await screen.findByText('Connection lost')).toBeDefined()
  })

  it('calls onComplete callback', async () => {
    const onComplete = vi.fn()
    mockGetDownloadStatus.mockResolvedValue({ ...baseProgress, status: 'complete', percentage: 100 })
    render(<DownloadManager modelId="gpt2" onComplete={onComplete} />)
    fireEvent.click(screen.getByText('Download model'))
    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled()
    })
  })
})
