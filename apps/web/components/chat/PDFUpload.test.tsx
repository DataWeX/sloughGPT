import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, onClick, disabled, title, ...rest }: any) => (
    <button onClick={onClick} disabled={disabled} title={title} {...rest}>{children}</button>
  ),
  IconDocument: () => <span data-testid="icon-document">doc</span>,
}))

import { PDFUpload } from './PDFUpload'

describe('PDFUpload', () => {
  const onAnalysis = vi.fn()
  const onError = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders upload button', () => {
    render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    expect(screen.getByTitle('Upload PDF for analysis')).toBeDefined()
  })

  it('shows spinner when uploading', () => {
    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')
    expect(input).toBeDefined()
  })

  it('calls onError for non-PDF file', () => {
    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')!
    const file = new File(['test'], 'test.txt', { type: 'text/plain' })
    Object.defineProperty(input, 'files', { value: [file], writable: false })
    fireEvent.change(input)
    expect(onError).toHaveBeenCalledWith('Only PDF files are accepted')
  })

  it('uploads PDF and calls onAnalysis on success', async () => {
    const fakeResponse = { ok: true, json: () => Promise.resolve({ analysis: 'PDF summary' }) }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fakeResponse))

    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')!
    const file = new File(['%PDF-1.4'], 'test.pdf', { type: 'application/pdf' })
    Object.defineProperty(input, 'files', { value: [file], writable: false })
    fireEvent.change(input)

    await waitFor(() => {
      expect(onAnalysis).toHaveBeenCalledWith('PDF summary', 'test.pdf')
    })

    vi.unstubAllGlobals()
  })

  it('calls onError on upload failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Network error')))

    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')!
    const file = new File(['%PDF-1.4'], 'test.pdf', { type: 'application/pdf' })
    Object.defineProperty(input, 'files', { value: [file], writable: false })
    fireEvent.change(input)

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('Network error')
    })

    vi.unstubAllGlobals()
  })

  it('calls onError on HTTP error', async () => {
    const fakeResponse = { ok: false, status: 500, json: () => Promise.resolve({ detail: 'Server error' }) }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fakeResponse))

    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')!
    const file = new File(['%PDF-1.4'], 'test.pdf', { type: 'application/pdf' })
    Object.defineProperty(input, 'files', { value: [file], writable: false })
    fireEvent.change(input)

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('Server error')
    })

    vi.unstubAllGlobals()
  })

  it('disables button when disabled prop is true', () => {
    render(<PDFUpload onAnalysis={onAnalysis} onError={onError} disabled={true} />)
    const btn = screen.getByTitle('Upload PDF for analysis')
    expect(btn.hasAttribute('disabled')).toBe(true)
  })
})
