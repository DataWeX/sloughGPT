import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import React from 'react'

const { mockPreview } = vi.hoisted(() => ({
  mockPreview: vi.fn(),
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: { preview: mockPreview },
  DatasetPreview: vi.fn(),
}))

import { DatasetPreview } from './DatasetPreview'

const sampleData = {
  total_samples: 1500,
  total_chars: 500_000,
  languages: { python: 800, javascript: 500, typescript: 200 },
  samples: [
    { path: 'main.py', language: 'python', content: 'print("hello")' },
    { path: 'app.js', language: 'javascript', content: 'console.log("hi")' },
  ],
}

describe('DatasetPreview', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('shows loading state', () => {
    mockPreview.mockReturnValue(new Promise(() => {}))
    render(<DatasetPreview datasetId="test" />)
    expect(screen.getByText('Loading preview...')).toBeDefined()
  })

  it('shows error state', async () => {
    mockPreview.mockRejectedValue(new Error('server error'))
    render(<DatasetPreview datasetId="test" />)
    await waitFor(() => {
      expect(screen.getByText('server error')).toBeDefined()
    })
  })

  it('renders preview stats', async () => {
    mockPreview.mockResolvedValue(sampleData)
    render(<DatasetPreview datasetId="test" />)
    await waitFor(() => {
      expect(screen.getByText('1500')).toBeDefined()
      expect(screen.getByText('488.3K')).toBeDefined()
      expect(screen.getByText('3')).toBeDefined()
    })
  })

  it('shows total files badge', async () => {
    mockPreview.mockResolvedValue(sampleData)
    render(<DatasetPreview datasetId="test" />)
    await waitFor(() => {
      expect(screen.getByText(/files/)).toBeDefined()
    })
  })

  it('shows top language', async () => {
    mockPreview.mockResolvedValue(sampleData)
    render(<DatasetPreview datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('python').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows language distribution badges', async () => {
    mockPreview.mockResolvedValue(sampleData)
    render(<DatasetPreview datasetId="test" />)
    await waitFor(() => {
      expect(screen.getByText('python: 800')).toBeDefined()
      expect(screen.getByText('javascript: 500')).toBeDefined()
    })
  })

  it('renders samples list', async () => {
    mockPreview.mockResolvedValue(sampleData)
    render(<DatasetPreview datasetId="test" />)
    await waitFor(() => {
      expect(screen.getByText('main.py')).toBeDefined()
      expect(screen.getByText('app.js')).toBeDefined()
    })
  })

  it('shows Content tab with textarea', async () => {
    mockPreview.mockResolvedValue(sampleData)
    const user = userEvent.setup()
    render(<DatasetPreview datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('Samples').length).toBeGreaterThanOrEqual(1)
    })
    await user.click(screen.getByRole('tab', { name: 'Content' }))
    await waitFor(() => {
      expect(screen.getByLabelText('Full dataset content preview')).toBeDefined()
    })
  })

  it('shows Use for Training button when callback provided', async () => {
    mockPreview.mockResolvedValue(sampleData)
    const onUse = vi.fn()
    render(<DatasetPreview datasetId="test" onUseForTraining={onUse} />)
    await waitFor(() => {
      fireEvent.click(screen.getByText('Use for Training'))
    })
    expect(onUse).toHaveBeenCalled()
  })
})
