import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import VqaSection from './VqaSection'

afterEach(cleanup)

function makeImageFile(): File {
  return new File(['fake'], 'cat.png', { type: 'image/png' })
}

describe('VqaSection', () => {
  it('renders the section title and question input', () => {
    render(<VqaSection asking={false} onAsk={vi.fn()} />)
    expect(screen.getByText('Visual Question Answering')).toBeTruthy()
    expect(screen.getByLabelText('Question')).toBeTruthy()
  })

  it('disables ask until an image and a question are provided', () => {
    render(<VqaSection asking={false} onAsk={vi.fn()} />)
    const button = screen.getByRole('button', { name: 'Ask' })
    expect(button).toHaveProperty('disabled', true)

    fireEvent.change(screen.getByLabelText('VQA image'), { target: { files: [makeImageFile()] } })
    fireEvent.change(screen.getByLabelText('Question'), {
      target: { value: 'What is in this image?' },
    })
    expect(button).toHaveProperty('disabled', false)
  })

  it('calls onAsk with the file and trimmed question', () => {
    const onAsk = vi.fn()
    const file = makeImageFile()
    render(<VqaSection asking={false} onAsk={onAsk} />)

    fireEvent.change(screen.getByLabelText('VQA image'), { target: { files: [file] } })
    fireEvent.change(screen.getByLabelText('Question'), {
      target: { value: '  What is in this image?  ' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }))

    expect(onAsk).toHaveBeenCalledWith(file, 'What is in this image?')
  })

  it('shows the answer when present', () => {
    render(
      <VqaSection
        asking={false}
        onAsk={vi.fn()}
        answer={{
          answer: 'A cat in a spacesuit',
          question: 'What is in this image?',
          elapsed_ms: 12.3,
        }}
      />,
    )
    expect(screen.getByTestId('vqa-answer').textContent).toContain('A cat in a spacesuit')
  })
})
