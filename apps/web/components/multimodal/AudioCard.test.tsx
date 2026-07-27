import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

const mockOnTranscribe = vi.fn()
const mockOnSynthesize = vi.fn()

import AudioCard from './AudioCard'

describe('AudioCard', () => {
  afterEach(cleanup)

  it('renders card title', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Audio')).toBeDefined()
  })

  it('shows Speech-to-text section', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Speech-to-text')).toBeDefined()
  })

  it('shows Text-to-speech section', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Text-to-speech')).toBeDefined()
  })

  it('renders Upload audio button', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Upload audio')).toBeDefined()
  })

  it('shows Transcribing... when transcribing is true', () => {
    render(<AudioCard transcribing={true} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Transcribing…')).toBeDefined()
  })

  it('disables Upload button when transcribing', () => {
    render(<AudioCard transcribing={true} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    const btn = screen.getByText('Transcribing…').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('displays transcript text when provided', () => {
    render(<AudioCard transcribing={false} transcript="Hello world" synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Hello world')).toBeDefined()
  })

  it('does not display transcript area when null', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.queryByText('Hello world')).toBeNull()
  })

  it('renders text-to-speech input', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByPlaceholderText('Text to speak…')).toBeDefined()
  })

  it('renders Speak button', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Speak')).toBeDefined()
  })

  it('Speak button disabled when text is empty', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    const btn = screen.getByText('Speak').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('Speak button enabled when text is entered', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    fireEvent.change(screen.getByPlaceholderText('Text to speak…'), { target: { value: 'hi' } })
    const btn = screen.getByText('Speak').closest('button')!
    expect(btn.disabled).toBe(false)
  })

  it('calls onSynthesize when Speak clicked', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    fireEvent.change(screen.getByPlaceholderText('Text to speak…'), { target: { value: 'speak this' } })
    fireEvent.click(screen.getByText('Speak'))
    expect(mockOnSynthesize).toHaveBeenCalledWith('speak this')
  })

  it('calls onSynthesize on Enter key in input', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={false} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    fireEvent.change(screen.getByPlaceholderText('Text to speak…'), { target: { value: 'enter text' } })
    fireEvent.keyDown(screen.getByPlaceholderText('Text to speak…'), { key: 'Enter' })
    expect(mockOnSynthesize).toHaveBeenCalledWith('enter text')
  })

  it('shows Synthesizing... when synthesizing is true', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={true} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    expect(screen.getByText('Synthesizing…')).toBeDefined()
  })

  it('Speak button disabled when synthesizing', () => {
    render(<AudioCard transcribing={false} transcript={null} synthesizing={true} onTranscribe={mockOnTranscribe} onSynthesize={mockOnSynthesize} />)
    fireEvent.change(screen.getByPlaceholderText('Text to speak…'), { target: { value: 'hi' } })
    const btn = screen.getByText('Synthesizing…').closest('button')!
    expect(btn.disabled).toBe(true)
  })
})
