import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, onClick, disabled, title, ...rest }: any) => (
    <button onClick={onClick} disabled={disabled} title={title} {...rest}>{children}</button>
  ),
  IconDocument: () => <span data-testid="icon-document">doc</span>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

const mockUploadPDF = vi.fn()

vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: {
    uploadPDF: (...args: unknown[]) => mockUploadPDF(...args),
  },
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
    mockUploadPDF.mockResolvedValue({ analysis: 'PDF summary' })

    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')!
    const file = new File(['%PDF-1.4'], 'test.pdf', { type: 'application/pdf' })
    Object.defineProperty(input, 'files', { value: [file], writable: false })
    fireEvent.change(input)

    await waitFor(() => {
      expect(onAnalysis).toHaveBeenCalledWith('PDF summary', 'test.pdf')
    })
  })

  it('calls onError on upload failure', async () => {
    mockUploadPDF.mockRejectedValue(new Error('Network error'))

    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')!
    const file = new File(['%PDF-1.4'], 'test.pdf', { type: 'application/pdf' })
    Object.defineProperty(input, 'files', { value: [file], writable: false })
    fireEvent.change(input)

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('Network error')
    })
  })

  it('calls onError on HTTP error', async () => {
    mockUploadPDF.mockRejectedValue(new Error('Server error'))

    const { container } = render(<PDFUpload onAnalysis={onAnalysis} onError={onError} />)
    const input = container.querySelector('input[type="file"]')!
    const file = new File(['%PDF-1.4'], 'test.pdf', { type: 'application/pdf' })
    Object.defineProperty(input, 'files', { value: [file], writable: false })
    fireEvent.change(input)

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('Server error')
    })
  })

  it('disables button when disabled prop is true', () => {
    render(<PDFUpload onAnalysis={onAnalysis} onError={onError} disabled={true} />)
    const btn = screen.getByTitle('Upload PDF for analysis')
    expect(btn.hasAttribute('disabled')).toBe(true)
  })
})
