import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

// ── cva mock ──
const mockCva = vi.hoisted(() => { const fn = () => ''; return fn })
vi.mock('class-variance-authority', () => ({ cva: () => mockCva }))

// ── controller & router mocks ──
const { mockGet, mockUpdate, mockDelete, mockExport, mockGetStats, mockPush, mockAddToast } = vi.hoisted(() => ({
  mockGet: vi.fn(), mockUpdate: vi.fn(), mockDelete: vi.fn(),
  mockExport: vi.fn(), mockGetStats: vi.fn(), mockPush: vi.fn(), mockAddToast: vi.fn(),
}))
const stableRouter = { push: mockPush }

vi.mock('next/navigation', () => ({ useRouter: () => stableRouter, useParams: () => ({ id: 'shakespeare' }) }))
vi.mock('@/lib/dataset-controller', () => ({ datasetController: { get: mockGet, update: mockUpdate, delete: mockDelete, export: mockExport, getStats: mockGetStats } }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/components/DatasetPreview', () => ({ DatasetPreview: () => null }))
vi.stubGlobal('confirm', vi.fn(() => true))
vi.stubGlobal('URL', { createObjectURL: vi.fn(), revokeObjectURL: vi.fn() })

import DatasetDetailPage from './page'

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockGetStats.mockResolvedValue({
    format: 'jsonl',
    samples: 500,
    chars: 120000,
    avg_length: 240,
    has_messages: false,
    sample_preview: [],
    lines: 500,
    suggested_method: 'finetune',
    file_type: 'jsonl',
  })
})

const mockDataset = {
  id: 'shakespeare', name: 'Shakespeare Works', type: 'text',
  source: 'local', size: 1048576, samples: 500,
  tags: ['literature'], created_at: '2026-06-01T12:00:00Z',
}

function waitForName() {
  return waitFor(() => { expect(screen.getAllByText('Shakespeare Works').length).toBeGreaterThan(0) })
}

describe('DatasetDetailPage', () => {

  it('shows loading initially and calls get', () => {
    mockGet.mockReturnValue(new Promise(() => {}))
    render(<DatasetDetailPage />)
    expect(screen.getByText('...')).toBeTruthy()
    expect(mockGet).toHaveBeenCalledWith('shakespeare')
  })

  it('displays dataset name after loading', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
  })

  it('shows not-found card on fetch failure', async () => {
    mockGet.mockRejectedValueOnce(new Error('not found'))
    render(<DatasetDetailPage />)
    await waitFor(() => { expect(screen.getByText('Dataset not found')).toBeTruthy() })
  })

  it('shows stat cards with correct values', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitFor(() => { expect(screen.getByText('text')).toBeTruthy() })
    expect(screen.getAllByText('500').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('1.0 MB')).toBeTruthy()
    expect(screen.getByText('local')).toBeTruthy()
    expect(screen.getByText('literature')).toBeTruthy()
    expect(screen.getByText('Preview')).toBeTruthy()
  })

  it('shows rename input on rename button click', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Rename dataset').click() })
    expect(screen.getByDisplayValue('Shakespeare Works')).toBeTruthy()
  })

  it('commits rename on confirm', async () => {
    mockUpdate.mockResolvedValue({})
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Rename dataset').click() })
    const input = screen.getByDisplayValue('Shakespeare Works')
    await act(async () => { fireEvent.change(input, { target: { value: 'Renamed' } }) })
    await act(async () => { screen.getByLabelText('Confirm rename').click() })
    await waitFor(() => { expect(mockUpdate).toHaveBeenCalledWith('shakespeare', { name: 'Renamed' }) })
  })

  it('cancels rename on cancel', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Rename dataset').click() })
    await act(async () => { screen.getByLabelText('Cancel rename').click() })
    expect(mockUpdate).not.toHaveBeenCalled()
  })

  it('deletes dataset on delete with confirmation', async () => {
    mockDelete.mockResolvedValue({})
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByText('Delete').click() })
    await waitFor(() => { expect(mockDelete).toHaveBeenCalledWith('shakespeare') })
    expect(mockPush).toHaveBeenCalledWith('/datasets')
  })

  it('exports dataset on export button click', async () => {
    mockExport.mockResolvedValue(new Blob())
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByText('Export').click() })
    await waitFor(() => { expect(mockExport).toHaveBeenCalledWith('shakespeare') })
  })

  it('fetches and displays dataset stats', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitFor(() => { expect(screen.getByText('Stats')).toBeTruthy() })
    expect(mockGetStats).toHaveBeenCalledWith('shakespeare')
    expect(screen.getByText('jsonl')).toBeTruthy()
    expect(screen.getAllByText('500').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('240 chars')).toBeTruthy()
    expect(screen.getByText('120,000')).toBeTruthy()
    expect(screen.getByText('finetune')).toBeTruthy()
  })

  it('does not show Stats card when stats fail', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockGetStats.mockRejectedValueOnce(new Error('stats unavailable'))
    render(<DatasetDetailPage />)
    await waitForName()
    await waitFor(() => { expect(mockGetStats).toHaveBeenCalledWith('shakespeare') })
    expect(screen.queryByText('Stats')).toBeFalsy()
  })
})
