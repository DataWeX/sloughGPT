// @vitest-environment jsdom
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ReadFileSection from './ReadFileSection'

describe('ReadFileSection', () => {
  const defaultProps = {
    readLoading: false,
    readFileData: null,
    onFileSelected: vi.fn(),
    onRemove: vi.fn(),
  }

  it('renders upload area when no file data', () => {
    render(<ReadFileSection {...defaultProps} />)
    expect(screen.getByText(/Drop a file here or click to upload/)).toBeTruthy()
    expect(screen.getByText(/PDF, Word, TXT, MD, CSV, JSON/)).toBeTruthy()
  })

  it('shows loading text when readLoading and no file data', () => {
    render(<ReadFileSection {...defaultProps} readLoading={true} />)
    expect(screen.getByText('Reading your file...')).toBeTruthy()
  })

  it('renders file info when file data provided', () => {
    const data = { text: 'hello', filename: 'test.pdf', pages: 5 }
    render(<ReadFileSection {...defaultProps} readFileData={data} />)
    expect(screen.getByText('test.pdf')).toBeTruthy()
    expect(screen.getByText('(5 pages)')).toBeTruthy()
  })

  it('does not show page count when pages is 1', () => {
    const data = { text: 'hello', filename: 'test.txt', pages: 1 }
    render(<ReadFileSection {...defaultProps} readFileData={data} />)
    expect(screen.queryByText('(1 pages)')).toBeNull()
  })

  it('calls onRemove when Remove button clicked', () => {
    const onRemove = vi.fn()
    const data = { text: 'hello', filename: 'test.pdf', pages: 1 }
    const { container } = render(<ReadFileSection {...defaultProps} readFileData={data} onRemove={onRemove} />)
    const btn = container.querySelector('button')
    expect(btn).toBeTruthy()
    fireEvent.click(btn!)
    expect(onRemove).toHaveBeenCalledOnce()
  })

  it('calls onFileSelected when file input changes', () => {
    const onFileSelected = vi.fn()
    const { container } = render(<ReadFileSection {...defaultProps} onFileSelected={onFileSelected} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    expect(input).toBeTruthy()
    const file = new File(['content'], 'test.txt', { type: 'text/plain' })
    const dataTransfer = { files: [file] }
    Object.defineProperty(input, 'files', { value: dataTransfer.files, configurable: true })
    fireEvent.change(input)
    expect(onFileSelected).toHaveBeenCalledOnce()
  })

  it('shows page count only when pages > 1', () => {
    const data = { text: 'hello', filename: 'doc.pdf', pages: 5 }
    const { container } = render(<ReadFileSection {...defaultProps} readFileData={data} />)
    expect(container.textContent).toContain('5 pages')
  })

  it('renders file name in filename display', () => {
    const data = { text: 'content', filename: 'report.csv', pages: 3 }
    render(<ReadFileSection {...defaultProps} readFileData={data} />)
    expect(screen.getByText('report.csv')).toBeTruthy()
  })
})
