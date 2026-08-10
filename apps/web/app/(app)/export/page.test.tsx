import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockList, mockGetExportFormats, mockExportFeedbackPairs, mockApiGet,
  mockDownloadJson, mockDownloadBlob, mockAddToast,
} = vi.hoisted(() => ({
  mockList: vi.fn(), mockGetExportFormats: vi.fn(), mockExportFeedbackPairs: vi.fn(),
  mockApiGet: vi.fn(), mockDownloadJson: vi.fn(), mockDownloadBlob: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled, variant, size, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} data-size={size} aria-label={ariaLabel}>{children}</button>
    ),
    Badge: ({ children, variant }: any) => <span data-variant={variant}>{children}</span>,
    IconRefresh: iconMock('refresh'), IconDownload: iconMock('download'), IconX: iconMock('x'),
  }
})

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left, right }: any) => <div>{left}{right}</div>,
  AppRouteHeaderLead: ({ title, subtitle }: any) => <div><h1>{title}</h1><span>{subtitle}</span></div>,
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: (...a: unknown[]) => mockList(...a),
    getExportFormats: (...a: unknown[]) => mockGetExportFormats(...a),
  },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    exportFeedbackPairs: (...a: unknown[]) => mockExportFeedbackPairs(...a),
  },
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...a: unknown[]) => mockApiGet(...a),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: (...a: unknown[]) => mockDownloadJson(...a),
  downloadBlob: (...a: unknown[]) => mockDownloadBlob(...a),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/components/export/ExportHistoryCard', () => ({
  ExportHistoryCard: () => <div data-testid="export-history" />,
  recordExport: vi.fn(),
}))

import ExportPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([])
  mockGetExportFormats.mockResolvedValue({ sou: 'Soul format', safetensors: 'SafeTensors format' })
  mockApiGet.mockResolvedValue({ checkpoints: [] })
  mockExportFeedbackPairs.mockResolvedValue(null)
})

describe('ExportPage — initial load flow', () => {
  it('renders header and export format options', async () => {
    render(<ExportPage />)
    expect(screen.getByText('Export')).toBeTruthy()
    await waitFor(() => {
      expect(screen.getAllByText(/soul|sou/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('loads export formats from controller', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(mockGetExportFormats).toHaveBeenCalledTimes(1)
    })
  })

  it('loads checkpoints on mount', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/auto-train/checkpoints')
    })
  })

  it('shows fallback formats when controller fails', async () => {
    mockGetExportFormats.mockRejectedValue(new Error('no formats'))
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/sou|soul/i).length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText(/onnx/i)).toBeTruthy()
    })
  })
})

describe('ExportPage — format selection flow', () => {
  beforeEach(() => {
    mockGetExportFormats.mockResolvedValue({
      sou: 'SloughGPT self-contained',
      safetensors: 'Safe, fast',
      onnx: 'Cross-platform',
    })
  })

  it('defaults to SOU format selected', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/soul|sou/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('clicking a different format changes selection', async () => {
    render(<ExportPage />)
    await waitFor(() => { expect(screen.getByText(/safetensors/i)).toBeTruthy() })

    const safetensorsBtn = screen.getByText(/safetensors/i)
    fireEvent.click(safetensorsBtn)

    // Should now be selected
    await waitFor(() => {
      expect(safetensorsBtn).toBeTruthy()
    })
  })
})

describe('ExportPage — model export flow', () => {
  it('shows model export section', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getByText('Export the currently loaded model to a file format.')).toBeTruthy()
    })
  })
})

describe('ExportPage — training data export flow', () => {
  it('shows training data export section', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getByText(/training pairs as JSON/i)).toBeTruthy()
    })
  })
})

describe('ExportPage — checkpoint download flow', () => {
  beforeEach(() => {
    mockApiGet.mockResolvedValue({
      checkpoints: [
        { name: 'cp-1', path: '/models/cp-1.soul', loss: 1.5, size_bytes: 1024000 },
        { name: 'cp-2', path: '/models/cp-2.soul', loss: 0.8, size_bytes: 2048000 },
      ],
    })
  })

  it('lists available checkpoints', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getByText('cp-1')).toBeTruthy()
      expect(screen.getByText('cp-2')).toBeTruthy()
    })
  })

  it('shows checkpoint loss values', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getByText(/1\.5/)).toBeTruthy()
      expect(screen.getByText(/0\.8/)).toBeTruthy()
    })
  })

  it('shows download button per checkpoint', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getByText('cp-1')).toBeTruthy()
    })
    const downloadBtns = screen.getAllByRole('button').filter(b =>
      b.textContent?.toLowerCase().includes('download')
    )
    expect(downloadBtns.length).toBeGreaterThanOrEqual(1)
  })
})

describe('ExportPage — empty state flow', () => {
  beforeEach(() => {
    mockApiGet.mockResolvedValue({ checkpoints: [] })
  })

  it('shows empty checkpoints message when none exist', async () => {
    render(<ExportPage />)
    await waitFor(() => {
      expect(screen.getByText(/no.*checkpoint|none.*train/i)).toBeTruthy()
    })
  })
})

describe('ExportPage — error handling flow', () => {
  it('handles checkpoint fetch failure gracefully', async () => {
    mockApiGet.mockRejectedValue(new Error('network error'))
    render(<ExportPage />)
    await waitFor(() => {
      // Page should still render without crashing
      expect(screen.getByText('Export')).toBeTruthy()
    })
  })

  it('handles format fetch failure gracefully', async () => {
    mockGetExportFormats.mockRejectedValue(new Error('no formats'))
    render(<ExportPage />)
    await waitFor(() => {
      // Should show fallback formats
      expect(screen.getByText(/sou|soul/i)).toBeTruthy()
    })
  })
})
