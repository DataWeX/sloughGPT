import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

import ImageGenerationCard from '@/components/multimodal/ImageGenerationCard'

afterEach(cleanup)

describe('ImageGenerationCard', () => {
  const onGenerate = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders title', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    expect(screen.getAllByText('Image Generation').length).toBeGreaterThanOrEqual(1)
  })

  it('renders prompt input', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    expect(screen.getAllByLabelText('Image generation prompt').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Generate button', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    expect(screen.getAllByText('Generate').length).toBeGreaterThanOrEqual(1)
  })

  it('disables Generate when no prompt', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    const btns = screen.getAllByText('Generate').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).toBeDisabled()
  })

  it('enables Generate with prompt', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    fireEvent.change(screen.getAllByLabelText('Image generation prompt')[0], { target: { value: 'A cat' } })
    const btns = screen.getAllByText('Generate').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).not.toBeDisabled()
  })

  it('calls onGenerate with prompt', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    fireEvent.change(screen.getAllByLabelText('Image generation prompt')[0], { target: { value: 'A sunset' } })
    const btns = screen.getAllByText('Generate').filter(el => el.closest('button'))
    fireEvent.click(btns[0].closest('button')!)
    expect(onGenerate).toHaveBeenCalledWith('A sunset')
  })

  it('trims whitespace', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    fireEvent.change(screen.getAllByLabelText('Image generation prompt')[0], { target: { value: '  hello  ' } })
    const btns = screen.getAllByText('Generate').filter(el => el.closest('button'))
    fireEvent.click(btns[0].closest('button')!)
    expect(onGenerate).toHaveBeenCalledWith('hello')
  })

  it('shows Generating when generating', () => {
    render(<ImageGenerationCard generating={true} onGenerate={onGenerate} />)
    expect(screen.getAllByText('Generating…').length).toBeGreaterThanOrEqual(1)
  })

  it('disables button when generating', () => {
    render(<ImageGenerationCard generating={true} onGenerate={onGenerate} />)
    const btns = screen.getAllByText('Generating…').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).toBeDisabled()
  })

  it('shows generated image', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} generatedImage="https://example.com/img.png" />)
    const imgs = screen.getAllByRole('img')
    expect(imgs.some(img => img.getAttribute('src') === 'https://example.com/img.png')).toBe(true)
  })

  it('does not show image when null', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} generatedImage={null} />)
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })

  it('Enter triggers generate', () => {
    render(<ImageGenerationCard generating={false} onGenerate={onGenerate} />)
    fireEvent.change(screen.getAllByLabelText('Image generation prompt')[0], { target: { value: 'A cat' } })
    fireEvent.keyDown(screen.getAllByLabelText('Image generation prompt')[0], { key: 'Enter' })
    expect(onGenerate).toHaveBeenCalledWith('A cat')
  })
})
