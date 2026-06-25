// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, variant, size, ...rest }: any) => (
    <button onClick={onClick} disabled={disabled} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
}))

vi.mock('@/components/ui/tags', () => ({
  Chip: ({ label, onClick }: any) => <button onClick={onClick} data-testid="chip">{label}</button>,
}))

import { PDFViewer } from './PDFViewer'

const samplePages = ['data:image/png;base64,page1', 'data:image/png;base64,page2', 'data:image/png;base64,page3']

describe('PDFViewer', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('returns null when pages is empty', () => {
    const { container } = render(<PDFViewer pages={[]} filename="doc.pdf" />)
    expect(container.innerHTML).toBe('')
  })

  it('renders filename and page count', () => {
    render(<PDFViewer pages={samplePages} filename="report.pdf" />)
    expect(screen.getByText('report.pdf')).toBeDefined()
    expect(screen.getByText('(3 pages)')).toBeDefined()
  })

  it('renders singular page label', () => {
    render(<PDFViewer pages={['page1']} filename="doc.pdf" />)
    expect(screen.getByText('(1 page)')).toBeDefined()
  })

  it('shows page image', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    const imgs = screen.getAllByRole('img')
    expect(imgs.some(img => img.getAttribute('src') === samplePages[0])).toBe(true)
  })

  it('shows pagination controls for multi-page', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    expect(screen.getByLabelText('Previous page')).toBeDefined()
    expect(screen.getByLabelText('Next page')).toBeDefined()
  })

  it('does not show pagination controls for single page', () => {
    render(<PDFViewer pages={['page1']} filename="doc.pdf" />)
    expect(screen.queryByLabelText('Previous page')).toBeNull()
    expect(screen.queryByLabelText('Next page')).toBeNull()
  })

  it('shows page counter', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    expect(screen.getByText('Page 1 of 3')).toBeDefined()
  })

  it('navigates to next and previous pages', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    fireEvent.click(screen.getByLabelText('Next page'))
    expect(screen.getByText('Page 2 of 3')).toBeDefined()
    fireEvent.click(screen.getByLabelText('Previous page'))
    expect(screen.getByText('Page 1 of 3')).toBeDefined()
  })

  it('disables Previous on first page', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    expect(screen.getByLabelText('Previous page').hasAttribute('disabled')).toBe(true)
  })

  it('disables Next on last page', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    fireEvent.click(screen.getByLabelText('Next page'))
    fireEvent.click(screen.getByLabelText('Next page'))
    expect(screen.getByText('Page 3 of 3')).toBeDefined()
    expect(screen.getByLabelText('Next page').hasAttribute('disabled')).toBe(true)
  })

  it('collapses/expands content on toggle', () => {
    const { container } = render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    expect(screen.getByText('Page 1 of 3')).toBeDefined()
    fireEvent.click(screen.getByLabelText('Collapse'))
    expect(screen.queryByText('Page 1 of 3')).toBeNull()
    fireEvent.click(screen.getByLabelText('Expand'))
    expect(screen.getByText('Page 1 of 3')).toBeDefined()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<PDFViewer pages={samplePages} filename="doc.pdf" onClose={onClose} />)
    fireEvent.click(screen.getByLabelText('Remove PDF'))
    expect(onClose).toHaveBeenCalled()
  })

  it('does not show close button when onClose not provided', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    expect(screen.queryByLabelText('Remove PDF')).toBeNull()
  })

  it('renders suggestions and calls onSuggestionClick', () => {
    const onSuggestionClick = vi.fn()
    render(
      <PDFViewer
        pages={samplePages}
        filename="doc.pdf"
        suggestions={['What is this about?', 'Summarize']}
        onSuggestionClick={onSuggestionClick}
      />
    )
    expect(screen.getByText('Try asking:')).toBeDefined()
    const chips = screen.getAllByTestId('chip')
    expect(chips.length).toBe(2)
    fireEvent.click(chips[0])
    expect(onSuggestionClick).toHaveBeenCalledWith('What is this about?')
  })

  it('does not render suggestions section without onSuggestionClick', () => {
    render(
      <PDFViewer pages={samplePages} filename="doc.pdf" suggestions={['What is this?']} />
    )
    expect(screen.queryByText('Try asking:')).toBeNull()
  })

  it('starts expanded by default', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" />)
    expect(screen.getByText('Page 1 of 3')).toBeDefined()
  })

  it('starts collapsed when defaultExpanded is false', () => {
    render(<PDFViewer pages={samplePages} filename="doc.pdf" defaultExpanded={false} />)
    expect(screen.queryByText('Page 1 of 3')).toBeNull()
  })
})
