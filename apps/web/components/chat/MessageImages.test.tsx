// @vitest-environment jsdom
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { MessageImages } from './MessageImages'

vi.mock('./ImageLightbox', () => ({
  ImageLightbox: ({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) => (
    <div data-testid="lightbox">
      <span>{alt}</span>
      <button onClick={onClose}>Close</button>
    </div>
  ),
}))

describe('MessageImages', () => {
  const images = [
    { id: '1', name: 'photo1.png', dataUrl: 'data:image/png;base64,abc1' },
    { id: '2', name: 'photo2.jpg', dataUrl: 'data:image/jpeg;base64,def2' },
  ]

  it('returns null when no images', () => {
    const { container } = render(<MessageImages images={[]} role="user" />)
    expect(container.innerHTML).toBe('')
  })

  it('returns null when images is undefined', () => {
    const { container } = render(<MessageImages images={undefined as any} role="user" />)
    expect(container.innerHTML).toBe('')
  })

  it('renders image thumbnails', () => {
    render(<MessageImages images={images} role="user" />)
    const imgs = screen.getAllByRole('img')
    expect(imgs.length).toBe(2)
    expect(imgs[0].getAttribute('alt')).toBe('photo1.png')
    expect(imgs[1].getAttribute('alt')).toBe('photo2.jpg')
  })

  it('opens lightbox when image clicked', () => {
    render(<MessageImages images={images} role="assistant" />)
    const buttons = screen.getAllByRole('button')
    fireEvent.click(buttons[0])
    expect(screen.getByTestId('lightbox')).toBeTruthy()
    expect(screen.getByText('Image preview')).toBeTruthy()
  })

  it('closes lightbox when close clicked', () => {
    render(<MessageImages images={images} role="assistant" />)
    fireEvent.click(screen.getAllByRole('button')[0])
    fireEvent.click(screen.getByText('Close'))
    expect(screen.queryByTestId('lightbox')).toBeNull()
  })

  it('applies flex-row-reverse for user role', () => {
    const { container } = render(<MessageImages images={images} role="user" />)
    expect(container.querySelector('.flex-row-reverse')).toBeTruthy()
  })

  it('does not apply flex-row-reverse for assistant role', () => {
    const { container } = render(<MessageImages images={images} role="assistant" />)
    expect(container.querySelector('.flex-row-reverse')).toBeNull()
  })
})
