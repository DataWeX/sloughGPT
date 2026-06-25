// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/ui', () => ({
  IconX: () => <span data-testid="icon-x">x</span>,
}))

import { ImageLightbox } from './ImageLightbox'

describe('ImageLightbox', () => {
  const onClose = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders image with given src and alt', () => {
    render(<ImageLightbox src="https://example.com/img.png" alt="test image" onClose={onClose} />)
    const img = screen.getByRole('img')
    expect(img).toBeDefined()
    expect(img.getAttribute('src')).toBe('https://example.com/img.png')
    expect(img.getAttribute('alt')).toBe('test image')
  })

  it('renders close button', () => {
    render(<ImageLightbox src="x.png" alt="x" onClose={onClose} />)
    expect(screen.getByTestId('icon-x')).toBeDefined()
  })

  it('calls onClose when backdrop clicked', () => {
    render(<ImageLightbox src="x.png" alt="x" onClose={onClose} />)
    const backdrop = screen.getByRole('dialog')
    fireEvent.click(backdrop)
    expect(onClose).toHaveBeenCalled()
  })

  it('does not call onClose when image clicked', () => {
    render(<ImageLightbox src="x.png" alt="x" onClose={onClose} />)
    const img = screen.getByRole('img')
    fireEvent.click(img)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('calls onClose on Escape key', () => {
    render(<ImageLightbox src="x.png" alt="x" onClose={onClose} />)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('sets body overflow hidden on mount', () => {
    render(<ImageLightbox src="x.png" alt="x" onClose={onClose} />)
    expect(document.body.style.overflow).toBe('hidden')
  })

  it('restores body overflow on unmount', () => {
    const { unmount } = render(<ImageLightbox src="x.png" alt="x" onClose={onClose} />)
    unmount()
    expect(document.body.style.overflow).toBe('')
  })

  it('has accessible role and label', () => {
    render(<ImageLightbox src="x.png" alt="x" onClose={onClose} />)
    const dialog = screen.getByRole('dialog')
    expect(dialog.getAttribute('aria-modal')).toBe('true')
    expect(dialog.getAttribute('aria-label')).toBe('Image preview')
  })
})
