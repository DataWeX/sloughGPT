import { describe, it, expect, vi, afterEach, beforeAll } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import ObjectDetectionSection from './ObjectDetectionSection'

beforeAll(() => {
  if (typeof URL.createObjectURL !== 'function') {
    URL.createObjectURL = vi.fn(() => 'blob:mock-preview')
    URL.revokeObjectURL = vi.fn()
  }
})

afterEach(cleanup)

function makeImageFile(): File {
  return new File(['fake'], 'scene.png', { type: 'image/png' })
}

const objects = [
  { label: 'cat', bbox: [0.1, 0.1, 0.5, 0.6], confidence: 0.72 },
  { label: 'window', bbox: [0.5, 0.0, 1.0, 0.4], confidence: 0.4 },
]

describe('ObjectDetectionSection', () => {
  it('renders the section title and image input', () => {
    render(<ObjectDetectionSection detecting={false} onDetect={vi.fn()} />)
    expect(screen.getByText('Object Detection')).toBeTruthy()
    expect(screen.getByLabelText('Object detection image')).toBeTruthy()
  })

  it('disables detect until an image is chosen', () => {
    render(<ObjectDetectionSection detecting={false} onDetect={vi.fn()} />)
    const button = screen.getByRole('button', { name: 'Detect' })
    expect(button).toHaveProperty('disabled', true)

    fireEvent.change(screen.getByLabelText('Object detection image'), { target: { files: [makeImageFile()] } })
    expect(button).toHaveProperty('disabled', false)
  })

  it('calls onDetect with the chosen file', () => {
    const onDetect = vi.fn()
    const file = makeImageFile()
    render(<ObjectDetectionSection detecting={false} onDetect={onDetect} />)

    fireEvent.change(screen.getByLabelText('Object detection image'), { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: 'Detect' }))

    expect(onDetect).toHaveBeenCalledWith(file)
  })

  it('renders detections as list rows and overlay boxes', () => {
    render(<ObjectDetectionSection detecting={false} onDetect={vi.fn()} objects={objects} />)
    expect(screen.getByTestId('detection-list').textContent).toContain('cat')
    expect(screen.getByTestId('detection-list').textContent).toContain('72%')

    fireEvent.change(screen.getByLabelText('Object detection image'), { target: { files: [makeImageFile()] } })
    expect(screen.getAllByTestId('detection-box')).toHaveLength(2)
  })

  it('renders nothing but the form when there are no detections', () => {
    render(<ObjectDetectionSection detecting={false} onDetect={vi.fn()} objects={[]} />)
    expect(screen.queryByTestId('detection-list')).toBeNull()
    expect(screen.queryByTestId('detection-preview')).toBeNull()
  })
})
