import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

import ImageTrainingCard from '@/components/multimodal/ImageTrainingCard'

describe('ImageTrainingCard', () => {
  const onUpload = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders title', () => {
    render(<ImageTrainingCard uploading={false} onUpload={onUpload} />)
    expect(screen.getAllByText('Image Training').length).toBeGreaterThanOrEqual(1)
  })

  it('renders description', () => {
    render(<ImageTrainingCard uploading={false} onUpload={onUpload} />)
    expect(screen.getAllByText(/Upload an image/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders upload button', () => {
    render(<ImageTrainingCard uploading={false} onUpload={onUpload} />)
    expect(screen.getAllByText('Upload image').length).toBeGreaterThanOrEqual(1)
  })

  it('has hidden file input', () => {
    render(<ImageTrainingCard uploading={false} onUpload={onUpload} />)
    const inputs = document.querySelectorAll('input[type="file"]')
    expect(inputs.length).toBeGreaterThanOrEqual(1)
    expect((inputs[0] as HTMLInputElement).accept).toBe('image/*')
  })

  it('triggers file input on button click', () => {
    render(<ImageTrainingCard uploading={false} onUpload={onUpload} />)
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const clickSpy = vi.spyOn(input, 'click')
    fireEvent.click(screen.getAllByText('Upload image')[0])
    expect(clickSpy).toHaveBeenCalled()
  })

  it('calls onUpload when file selected', () => {
    render(<ImageTrainingCard uploading={false} onUpload={onUpload} />)
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['dummy'], 'photo.png', { type: 'image/png' })
    Object.defineProperty(input, 'files', { value: [file] })
    fireEvent.change(input)
    expect(onUpload).toHaveBeenCalledWith(file)
  })

  it('shows Training when uploading', () => {
    render(<ImageTrainingCard uploading={true} onUpload={onUpload} />)
    expect(screen.getAllByText('Training…').length).toBeGreaterThanOrEqual(1)
  })

  it('disables button when uploading', () => {
    render(<ImageTrainingCard uploading={true} onUpload={onUpload} />)
    const btns = screen.getAllByText('Training…').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).toBeDisabled()
  })
})
