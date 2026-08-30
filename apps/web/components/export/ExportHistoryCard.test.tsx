// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn((key: string) => {
      const raw = localStorage.getItem(key)
      return Promise.resolve(raw ? JSON.parse(raw) : undefined)
    }),
    setKV: vi.fn((key: string, value: unknown) => {
      localStorage.setItem(key, JSON.stringify(value))
      return Promise.resolve()
    }),
    deleteKV: vi.fn((key: string) => {
      localStorage.removeItem(key)
      return Promise.resolve()
    }),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import { ExportHistoryCard, recordExport } from './ExportHistoryCard'

const STORAGE_KEY = 'sloughgpt-export-history'

afterEach(() => {
  cleanup()
  localStorage.removeItem(STORAGE_KEY)
})

beforeEach(() => {
  localStorage.removeItem(STORAGE_KEY)
})

describe('ExportHistoryCard', () => {
  it('renders empty state for empty history', async () => {
    const { container } = render(<ExportHistoryCard />)
    await waitFor(() => {})
    expect(container.innerHTML).toBe('')
  })

  it('renders when history exists', async () => {
    await recordExport('sou', 3)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByTestId('export-history').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Export History').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total exports', async () => {
    await recordExport('sou', 2)
    await recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows formats used count', async () => {
    await recordExport('sou', 1)
    await recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows total files', async () => {
    await recordExport('sou', 3)
    await recordExport('onnx', 2)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows last export time', async () => {
    await recordExport('sou', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText(/Last Export/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows recent entries', async () => {
    await recordExport('sou', 1)
    await recordExport('gguf_q4_k_m', 2)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Sou').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Gguf Q4 K M').length).toBeGreaterThanOrEqual(1)
  })

  it('shows file count per entry', async () => {
    await recordExport('sou', 3)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText(/3 files/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows singular file for count 1', async () => {
    await recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText(/1 file/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('recordExport persists to localStorage', async () => {
    await recordExport('sou', 2)
    const raw = localStorage.getItem(STORAGE_KEY)
    expect(raw).toBeTruthy()
    const arr = JSON.parse(raw!)
    expect(arr.length).toBe(1)
    expect(arr[0].format).toBe('sou')
    expect(arr[0].fileCount).toBe(2)
  })
})
