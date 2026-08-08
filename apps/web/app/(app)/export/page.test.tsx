import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockList = vi.fn()
const mockGetExportFormats = vi.fn()

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: (...args: unknown[]) => mockList(...args),
    getExportFormats: (...args: unknown[]) => mockGetExportFormats(...args),
  },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    exportFeedbackPairs: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn().mockResolvedValue(null),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
  downloadBlob: vi.fn(),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import ExportPage from './page'

describe('ExportPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockGetExportFormats.mockResolvedValue({ sou: 'Soul format', onnx: 'ONNX format' })
  })

  it('renders page header', async () => {
    render(<ExportPage />)
    expect(screen.getAllByText('Export').length).toBeGreaterThanOrEqual(1)
  })

  it('renders export format options after loading', async () => {
    render(<ExportPage />)
    await screen.findAllByText(/sou|soul|format/i)
    expect(screen.getAllByText(/sou|soul|format/i).length).toBeGreaterThanOrEqual(1)
  })

  it('shows export button', async () => {
    render(<ExportPage />)
    await screen.findAllByText(/export/i)
    expect(screen.getAllByText(/export/i).length).toBeGreaterThanOrEqual(1)
  })
})
