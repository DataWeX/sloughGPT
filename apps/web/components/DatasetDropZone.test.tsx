// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockUpload = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/files-controller', () => ({
  filesController: { upload: mockUpload },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  IconUpload: (props: any) => <svg data-testid="icon-upload" {...props} />,
}))

import { DatasetDropZone } from './DatasetDropZone'

afterEach(() => { cleanup(); vi.clearAllMocks() })

describe('DatasetDropZone', () => {
  it('renders without crashing', () => {
    render(<DatasetDropZone onUploadComplete={vi.fn()} />)
    expect(screen.getByText('Drag & drop files here')).toBeDefined()
    expect(screen.getByText(/or click to browse/)).toBeDefined()
  })

  it('has a hidden file input', () => {
    render(<DatasetDropZone onUploadComplete={vi.fn()} />)
    const input = screen.getByLabelText('Upload dataset files')
    expect(input).toBeDefined()
    expect(input).toHaveAttribute('type', 'file')
    expect(input).toHaveAttribute('multiple')
  })

  it('calls onUploadComplete after successful upload', async () => {
    mockUpload.mockResolvedValue(undefined)
    const onComplete = vi.fn()
    render(<DatasetDropZone onUploadComplete={onComplete} />)

    const input = screen.getByLabelText('Upload dataset files')
    const file = new File(['test'], 'test.jsonl', { type: 'text/plain' })
    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() => {
      expect(mockUpload).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith('Uploaded 1 file(s)', 'success')
      expect(onComplete).toHaveBeenCalled()
    })
  })

  it('shows error toast on upload failure', async () => {
    mockUpload.mockRejectedValue(new Error('fail'))
    render(<DatasetDropZone onUploadComplete={vi.fn()} />)

    const input = screen.getByLabelText('Upload dataset files')
    const file = new File(['test'], 'test.jsonl', { type: 'text/plain' })
    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Upload failed', 'error')
    })
  })
})
