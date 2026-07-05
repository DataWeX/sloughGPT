
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
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
    expect(screen.getByText('Drop image to attach')).toBeDefined()
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

  it('ignores non-image files on drop', () => {
    const onImageDropped = vi.fn()
    const file = new File(['hello'], 'test.txt', { type: 'text/plain' })
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div data-testid="child">Child content</div>
      </ImageDropZone>
    )
    const zone = screen.getByTestId('child').parentElement!
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(onImageDropped).not.toHaveBeenCalled()
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
    expect(screen.getByText('Drop image to attach')).toBeDefined()
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(screen.queryByText('Drop image to attach')).toBeNull()
  })
})
