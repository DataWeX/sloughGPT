// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/model-controller', () => ({
  modelController: { getExportFormats: vi.fn() },
}))
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    exportTrainingPairs: vi.fn(),
    downloadCheckpoint: vi.fn(),
  },
}))
vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))
vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
  downloadBlob: vi.fn(),
}))
vi.mock('@/components/export/ExportHistoryCard', () => ({
  ExportHistoryCard: () => <div data-testid="export-history" />,
  recordExport: vi.fn(),
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Badge: ({ children }: any) => <span>{children}</span>,
  IconDownload: () => <span>↓</span>,
  IconRefresh: () => <span>↻</span>,
}))

import ExportContent from './ExportContent'
import { modelController } from '@/lib/model-controller'
import { apiGet } from '@/lib/http-client'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(modelController.getExportFormats).mockResolvedValue([
    { key: 'sou', label: 'Soul (.soul)' },
    { key: 'gguf', label: 'GGUF' },
  ] as any)
  vi.mocked(apiGet).mockResolvedValue({ checkpoints: [{ name: 'cp1' }, { name: 'cp2' }] })
})

afterEach(() => cleanup())

describe('ExportContent', () => {
  it('renders section titles', async () => {
    render(<ExportContent />)
    expect(screen.getByText('Model Export')).toBeDefined()
    expect(screen.getByText('Training Data Export')).toBeDefined()
  })

  it('shows export history card', async () => {
    render(<ExportContent />)
    expect(screen.getByTestId('export-history')).toBeDefined()
  })

  it('calls getExportFormats on mount', async () => {
    render(<ExportContent />)
    await waitFor(() => {
      expect(modelController.getExportFormats).toHaveBeenCalled()
    })
  })
})
