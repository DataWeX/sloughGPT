// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import React from 'react'

import { ImageDropZone } from './ImageDropZone'

describe('ImageDropZone', () => {
  const onImageDropped = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders children', () => {
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div data-testid="child">content</div>
      </ImageDropZone>
    )
    expect(screen.getByTestId('child')).toBeDefined()
  })

  it('shows drop overlay on drag over with files', () => {
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div>content</div>
      </ImageDropZone>
    )
    const zone = screen.getByText('content').parentElement!
    const dataTransfer = { types: ['Files'] }
    fireEvent.dragOver(zone, { dataTransfer })
    expect(screen.getByText('Drop image to attach')).toBeDefined()
  })

  it('does not show overlay on drag over without files', () => {
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div>content</div>
      </ImageDropZone>
    )
    const zone = screen.getByText('content').parentElement!
    const dataTransfer = { types: ['text/plain'] }
    fireEvent.dragOver(zone, { dataTransfer })
    expect(screen.queryByText('Drop image to attach')).toBeNull()
  })

  it('hides overlay on drag leave', () => {
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div>content</div>
      </ImageDropZone>
    )
    const zone = screen.getByText('content').parentElement!
    const dataTransfer = { types: ['Files'] }
    fireEvent.dragOver(zone, { dataTransfer })
    expect(screen.getByText('Drop image to attach')).toBeDefined()
    fireEvent.dragLeave(zone)
    expect(screen.queryByText('Drop image to attach')).toBeNull()
  })

  it('calls onImageDropped with image files on drop', () => {
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div>content</div>
      </ImageDropZone>
    )
    const zone = screen.getByText('content').parentElement!
    const file = new File([''], 'test.png', { type: 'image/png' })
    const dataTransfer = { files: [file] }
    fireEvent.dragOver(zone, { dataTransfer: { types: ['Files'] } })
    fireEvent.drop(zone, { dataTransfer })
    expect(onImageDropped).toHaveBeenCalledWith(file)
  })

  it('filters non-image files on drop', () => {
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div>content</div>
      </ImageDropZone>
    )
    const zone = screen.getByText('content').parentElement!
    const textFile = new File([''], 'test.txt', { type: 'text/plain' })
    const dataTransfer = { files: [textFile] }
    fireEvent.drop(zone, { dataTransfer })
    expect(onImageDropped).not.toHaveBeenCalled()
  })

  it('hides overlay on drop', () => {
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div>content</div>
      </ImageDropZone>
    )
    const zone = screen.getByText('content').parentElement!
    const file = new File([''], 'test.png', { type: 'image/png' })
    fireEvent.dragOver(zone, { dataTransfer: { types: ['Files'] } })
    expect(screen.getByText('Drop image to attach')).toBeDefined()
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(screen.queryByText('Drop image to attach')).toBeNull()
  })

  it('auto-hides overlay after 3 seconds', async () => {
    vi.useFakeTimers()
    render(
      <ImageDropZone onImageDropped={onImageDropped}>
        <div>content</div>
      </ImageDropZone>
    )
    const zone = screen.getByText('content').parentElement!
    fireEvent.dragOver(zone, { dataTransfer: { types: ['Files'] } })
    expect(screen.getByText('Drop image to attach')).toBeDefined()
    act(() => { vi.advanceTimersByTime(3000) })
    expect(screen.queryByText('Drop image to attach')).toBeNull()
    vi.useRealTimers()
  })
})
