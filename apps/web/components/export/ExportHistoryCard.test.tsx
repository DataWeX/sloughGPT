// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
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
  it('renders empty state for empty history', () => {
    const { container } = render(<ExportHistoryCard />)
    expect(container.innerHTML).toBe('')
  })

  it('renders when history exists', () => {
    recordExport('sou', 3)
    render(<ExportHistoryCard />)
    expect(screen.getAllByTestId('export-history').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Export History').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total exports', () => {
    recordExport('sou', 2)
    recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows formats used count', () => {
    recordExport('sou', 1)
    recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total files', () => {
    recordExport('sou', 3)
    recordExport('onnx', 2)
    render(<ExportHistoryCard />)
    expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
  })

  it('shows last export time', () => {
    recordExport('sou', 1)
    render(<ExportHistoryCard />)
    expect(screen.getAllByText(/Last Export/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows recent entries', () => {
    recordExport('sou', 1)
    recordExport('gguf_q4_k_m', 2)
    render(<ExportHistoryCard />)
    expect(screen.getAllByText('Sou').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Gguf Q4 K M').length).toBeGreaterThanOrEqual(1)
  })

  it('shows file count per entry', () => {
    recordExport('sou', 3)
    render(<ExportHistoryCard />)
    expect(screen.getAllByText(/3 files/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows singular file for count 1', () => {
    recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    expect(screen.getAllByText(/1 file/).length).toBeGreaterThanOrEqual(1)
  })

  it('recordExport persists to localStorage', () => {
    recordExport('sou', 2)
    const raw = localStorage.getItem(STORAGE_KEY)
    expect(raw).toBeTruthy()
    const arr = JSON.parse(raw!)
    expect(arr.length).toBe(1)
    expect(arr[0].format).toBe('sou')
    expect(arr[0].fileCount).toBe(2)
  })
})
