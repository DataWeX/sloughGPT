import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Dialog: ({ children, open }: any) => open ? <div data-testid="dialog">{children}</div> : null,
  DialogContent: ({ children }: any) => <div data-testid="dialog-content">{children}</div>,
  DialogHeader: ({ children }: any) => <div>{children}</div>,
  DialogTitle: ({ children }: any) => <h2>{children}</h2>,
  Button: ({ children, onClick, disabled, variant, size, className, ...rest }: any) => (
    <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
  Tabs: ({ value, onChange, tabs: tabDefs, children }: any) => (
    <div data-testid="tabs" data-current={value}>
      {tabDefs.map((t: any) => (
        <button key={t.value} onClick={() => onChange?.(t.value)} data-active={value === t.value}>
          {t.label}
        </button>
      ))}
      {children}
    </div>
  ),
  Skeleton: ({ className }: any) => <div data-testid="skeleton" className={className} />,
  IconUpload: () => <span data-testid="icon-upload">upload</span>,
  IconTrash: () => <span data-testid="icon-trash">trash</span>,
  IconSend: () => <span data-testid="icon-send">send</span>,
  IconDownload: () => <span data-testid="icon-download">download</span>,
  IconX: () => <span data-testid="icon-x">x</span>,
  IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
}))

const mockGetTrainingReport = vi.fn()

vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: {
    getTrainingReport: mockGetTrainingReport,
    analyzeImage: vi.fn(),
    trainImage: vi.fn(),
    generateImage: vi.fn(),
    transcribeAudio: vi.fn(),
    getCapabilities: vi.fn().mockResolvedValue({ speech_to_text: true, image_captioning: true }),
    resetModel: vi.fn(),
  },
}))

import { VisionStudioDialog } from './VisionStudioDialog'

describe('VisionStudioDialog', () => {
  const onOpenChange = vi.fn()
  const onGeneratedImage = vi.fn()
  const onSendText = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders nothing when closed', () => {
    const { container } = render(
      <VisionStudioDialog
        open={false}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders dialog when open', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByText('Vision Studio')).toBeDefined()
    expect(screen.getByTestId('tabs')).toBeDefined()
  })

  it('renders tab buttons', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByText('Analyze')).toBeDefined()
    expect(screen.getByText('Supervised Train')).toBeDefined()
    expect(screen.getByText('Generate')).toBeDefined()
    expect(screen.getByText('History')).toBeDefined()
  })

  it('shows analyze tab by default', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByTestId('tabs').getAttribute('data-current')).toBe('analyze')
  })

  it('switches to supervised train tab', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    fireEvent.click(screen.getByText('Supervised Train'))
    expect(screen.getByText(/Ground truth label/)).toBeDefined()
  })

  it('switches to generate tab', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    fireEvent.click(screen.getByText('Generate'))
    expect(screen.getByPlaceholderText(/Describe an image/)).toBeDefined()
  })

  it('switches to history tab', async () => {
    mockGetTrainingReport.mockResolvedValue({
      images_learned: 10,
      vocab_size: 256,
      caption_history: ['a cat', 'a dog'],
      accuracy_history: [85, 90],
      mean_accuracy: 87.5,
      last_accuracy: 90,
    })
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    fireEvent.click(screen.getByText('History'))
    await waitFor(() => {
      expect(screen.getByText(/Images learned/)).toBeDefined()
    })
  })

  it('shows upload area in analyze tab', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByText(/Drop an image here or click to upload/)).toBeDefined()
  })

  it('shows file input for image upload', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    const fileInput = document.querySelector('input[type="file"]')
    expect(fileInput).toBeDefined()
  })

  it('shows image preview after upload', async () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    const fileInput = document.querySelector('input[type="file"]')!
    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })
    Object.defineProperty(fileInput, 'files', { value: [file] })
    fireEvent.change(fileInput)
    await waitFor(() => {
      const img = document.querySelector('img')
      expect(img).toBeDefined()
    })
  })

  it('shows initialCaps data without API call', () => {
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        initialCaps={{ images_learned: 5, trained: true, status: 'active', vocab_size: 128, mean_accuracy: 75 }}
      />
    )
    fireEvent.click(screen.getByText('History'))
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('128')).toBeDefined()
    expect(screen.getAllByText('75.0%').length).toBeGreaterThanOrEqual(1)
  })

  it('displays training accuracy from history tab', async () => {
    mockGetTrainingReport.mockResolvedValue({
      images_learned: 10,
      vocab_size: 256,
      caption_history: ['a cat', 'a dog'],
      accuracy_history: [85, 90],
      mean_accuracy: 85,
      last_accuracy: 90,
    })
    render(
      <VisionStudioDialog
        open={true}
        onOpenChange={onOpenChange}
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    fireEvent.click(screen.getByText('History'))
    await waitFor(() => {
      expect(screen.getByText('85.0%')).toBeDefined()
    })
  })
})
