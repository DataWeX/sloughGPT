
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import { ImageDropZone } from './ImageDropZone'

describe('ImageDropZone', () => {
  afterEach(cleanup)

  it('renders children', () => {
    render(
      <ImageDropZone onImageDropped={() => {}}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    expect(screen.getByTestId('child')).toBeDefined()
  })

  it('shows drop overlay on drag over with files', () => {
    render(
      <ImageDropZone onImageDropped={() => {}}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.dragOver(zone, { dataTransfer: { types: ['Files'] } })
    expect(screen.getByText('Drop files to attach')).toBeDefined()
  })

  it('calls onImageDropped for each image file on drop', () => {
    const onImageDropped = vi.fn()
    const file = new File(['hello'], 'test.png', { type: 'image/png' })
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(onImageDropped).toHaveBeenCalledWith(file)
  })

  it('falls back to onImageDropped for text files when no onTextDropped', () => {
    const onImageDropped = vi.fn()
    const file = new File(['hello'], 'test.txt', { type: 'text/plain' })
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(onImageDropped).toHaveBeenCalledWith(file)
  })

  it('hides overlay after drop', () => {
    const onImageDropped = vi.fn()
    const file = new File(['hello'], 'test.png', { type: 'image/png' })
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.dragOver(zone, { dataTransfer: { types: ['Files'] } })
    expect(screen.getByText('Drop files to attach')).toBeDefined()
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(screen.queryByText('Drop files to attach')).toBeNull()
  })

  it('calls onTextDropped for text files', async () => {
    const onTextDropped = vi.fn()
    const file = new File(['console.log("hello")'], 'app.js', { type: 'text/javascript' })
    render(
      <ImageDropZone onImageDropped={() => {}} onTextDropped={onTextDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    await waitFor(() => {
      expect(onTextDropped).toHaveBeenCalledWith('console.log("hello")', 'app.js')
    })
  })

  it('calls onPDFDropped for PDF files', () => {
    const onPDFDropped = vi.fn()
    const file = new File(['%PDF-1.4'], 'doc.pdf', { type: 'application/pdf' })
    render(
      <ImageDropZone onImageDropped={() => {}} onPDFDropped={onPDFDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(onPDFDropped).toHaveBeenCalledWith(file)
  })

  it('falls back to onImageDropped for unknown file types', () => {
    const onImageDropped = vi.fn()
    const file = new File(['data'], 'file.xyz', { type: 'application/octet-stream' })
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(onImageDropped).toHaveBeenCalledWith(file)
  })

  it('handles multiple files on drop', () => {
    const onImageDropped = vi.fn()
    const file1 = new File(['hello'], 'a.png', { type: 'image/png' })
    const file2 = new File(['world'], 'b.png', { type: 'image/png' })
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.drop(zone, { dataTransfer: { files: [file1, file2] } })
    expect(onImageDropped).toHaveBeenCalledTimes(2)
  })
})
