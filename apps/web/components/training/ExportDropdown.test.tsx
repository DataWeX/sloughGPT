import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const { mockDownload, mockAddToast } = vi.hoisted(() => ({
  mockDownload: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    downloadTrainingJob: mockDownload,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

import { ExportDropdown } from './ExportDropdown'

beforeEach(() => {
  vi.clearAllMocks()
  URL.createObjectURL = vi.fn(() => 'blob:test')
  URL.revokeObjectURL = vi.fn()
})

describe('ExportDropdown', () => {
  afterEach(cleanup)

  it('renders format selector with options', () => {
    render(<ExportDropdown jobId="job-1" checkpoint="model.pt" />)
    expect(screen.getByRole('combobox')).toBeDefined()
  })

  it('renders export button', () => {
    render(<ExportDropdown jobId="job-1" checkpoint="model.pt" />)
    expect(screen.getByText('Export')).toBeDefined()
  })

  it('changes button text during export', () => {
    mockDownload.mockReturnValue(new Promise(() => {}))
    render(<ExportDropdown jobId="job-1" checkpoint="model.pt" />)
    fireEvent.click(screen.getByText('Export'))
    expect(screen.getByText('Exporting...')).toBeDefined()
  })

  it('shows success toast on export', async () => {
    const fakeBlob = new Blob(['test'])
    Object.defineProperty(fakeBlob, 'size', { value: 4 })
    mockDownload.mockResolvedValue(fakeBlob)

    render(<ExportDropdown jobId="job-1" checkpoint="model.pt" />)
    fireEvent.click(screen.getByText('Export'))
    await screen.findByText('Export')
    expect(mockAddToast).toHaveBeenCalledWith(expect.stringContaining('Downloaded'), 'success')
  })

  it('shows error toast on export failure', async () => {
    mockDownload.mockRejectedValue(new Error('Network error'))
    render(<ExportDropdown jobId="job-1" checkpoint="model.pt" />)
    fireEvent.click(screen.getByText('Export'))
    await screen.findByText('Export')
    expect(mockAddToast).toHaveBeenCalledWith(expect.stringContaining('Export failed'), 'error')
  })
})
