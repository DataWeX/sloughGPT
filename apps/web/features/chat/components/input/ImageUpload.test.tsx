import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { ImageUpload, ImagePreview } from './ImageUpload'

describe('ImageUpload', () => {
  afterEach(cleanup)

  it('renders upload button', () => {
    render(<ImageUpload onImage={vi.fn()} />)
    const buttons = screen.getAllByLabelText('Upload image')
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('renders hidden file input', () => {
    render(<ImageUpload onImage={vi.fn()} />)
    const input = document.querySelector('input[type="file"]')
    expect(input).toBeDefined()
    expect(input?.getAttribute('accept')).toBe('image/*')
  })

  it('disables button when disabled', () => {
    render(<ImageUpload onImage={vi.fn()} disabled={true} />)
    const btns = screen.getAllByLabelText('Upload image')
    btns.forEach(b => expect(b.hasAttribute('disabled')).toBe(true))
  })
})

describe('ImagePreview', () => {
  const image = { id: 'img1', dataUrl: 'data:image/png;base64,abc', name: 'test.png' }

  afterEach(cleanup)

  it('renders image with alt text', () => {
    render(<ImagePreview image={image} onRemove={vi.fn()} />)
    const img = screen.getByAltText('test.png') as HTMLImageElement
    expect(img).toBeDefined()
    expect(img.src).toContain('data:image/png')
  })

  it('renders remove button', () => {
    render(<ImagePreview image={image} onRemove={vi.fn()} />)
    expect(screen.getByLabelText('Remove test.png')).toBeDefined()
  })

  it('calls onRemove when remove button clicked', () => {
    const onRemove = vi.fn()
    render(<ImagePreview image={image} onRemove={onRemove} />)
    fireEvent.click(screen.getByLabelText('Remove test.png'))
    expect(onRemove).toHaveBeenCalledWith('img1')
  })

  it('renders remove button with correct label', () => {
    render(<ImagePreview image={{ id: 'img2', dataUrl: 'data:image/png;base64,xyz', name: 'photo.jpg' }} onRemove={vi.fn()} />)
    expect(screen.getByLabelText('Remove photo.jpg')).toBeDefined()
  })

  it('renders image with correct dataUrl src', () => {
    render(<ImagePreview image={image} onRemove={vi.fn()} />)
    const img = screen.getByAltText('test.png') as HTMLImageElement
    expect(img.src).toBe('data:image/png;base64,abc')
  })
})
