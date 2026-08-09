import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'

vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: {
    transcribeAudio: vi.fn(),
    getCapabilities: vi.fn().mockResolvedValue({ speech_to_text: true }),
    resetModel: vi.fn(),
  },
}))

import { ChatInputAccessories } from './ChatInputAccessories'

describe('ChatInputAccessories', () => {
  const base = {
    onImage: vi.fn(),
    onTranscript: vi.fn(),
    disabled: false,
  }

  it('renders image upload button', () => {
    const { container } = render(<ChatInputAccessories {...base} />)
    const btns = container.querySelectorAll('button[title="Upload image"]')
    expect(btns.length >= 1).toBe(true)
  })

  it('renders audio upload button', () => {
    render(<ChatInputAccessories {...base} />)
    const btns = screen.getAllByRole('button')
    const audioBtn = btns.find(b => b.getAttribute('aria-label') === 'Upload audio')
    expect(audioBtn).toBeInTheDocument()
  })

  it('renders PDFUpload when callbacks provided', () => {
    render(<ChatInputAccessories {...base} onPDFAnalysis={vi.fn()} onPDFError={vi.fn()} />)
    const btns = screen.getAllByRole('button')
    expect(btns.some(b => b.title === 'Upload PDF for analysis')).toBe(true)
  })

  it('does not render PDFUpload when callbacks missing', () => {
    const { container } = render(<ChatInputAccessories {...base} />)
    expect(container.querySelector('[title="Upload PDF for analysis"]')).not.toBeInTheDocument()
  })

  it('passes disabled state', () => {
    const { container } = render(<ChatInputAccessories {...base} disabled />)
    const btns = container.querySelectorAll('button')
    btns.forEach(btn => {
      if (btn.getAttribute('title') !== 'Start voice input' && btn.getAttribute('title') !== 'Stop listening') {
        expect(btn).toBeDisabled()
      }
    })
  })
})
