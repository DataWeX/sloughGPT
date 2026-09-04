import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { AnalyzeTab } from './AnalyzeTab'
import type { UseVisionStudioReturn } from './useVisionStudio'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, onClick, disabled, variant, size, className, ...rest }: any) =>
    React.createElement('button', { onClick, disabled, className, ...rest }, children),
  IconUpload: (p: any) => React.createElement('svg', { 'data-testid': 'icon-upload', ...p }),
  IconX: (p: any) => React.createElement('svg', { 'data-testid': 'icon-x', ...p }),
  IconRefresh: (p: any) => React.createElement('svg', { 'data-testid': 'icon-refresh', ...p }),
  IconSend: (p: any) => React.createElement('svg', { 'data-testid': 'icon-send', ...p }),
  IconDownload: (p: any) => React.createElement('svg', { 'data-testid': 'icon-download', ...p }),
  IconAlert: (p: any) => React.createElement('svg', { 'data-testid': 'icon-alert', ...p }),
  Skeleton: ({ className }: any) => React.createElement('div', { 'data-testid': 'skeleton', className }),
}))

const mockVs = (overrides: Record<string, any> = {}): UseVisionStudioReturn => ({
  dropRef: { current: null } as any,
  fileInputRef: { current: null } as any,
  dragOver: false,
  setDragOver: vi.fn() as any,
  handleDrop: vi.fn() as any,
  handleFileSelect: vi.fn() as any,
  previewUrl: '',
  previewFileName: null,
  clearPreview: vi.fn() as any,
  analyzeLoading: false,
  analyzeError: '',
  analyzeResult: null,
  retryAnalyze: vi.fn() as any,
  retryLoading: false,
  tab: 'analyze',
  setTab: vi.fn() as any,
  trainLabel: '',
  setTrainLabel: vi.fn() as any,
  trainLoading: false,
  trainResult: null,
  genPrompt: '',
  setGenPrompt: vi.fn() as any,
  genLoading: false,
  genResult: null,
  genError: '',
  setGenResult: vi.fn() as any,
  trainingReport: null,
  resetLoading: false,
  refreshReport: vi.fn() as any,
  processFile: vi.fn() as any,
  handleTrainWithLabel: vi.fn() as any,
  handleGenerateImage: vi.fn() as any,
  handleSendGeneratedImage: vi.fn() as any,
  handleReset: vi.fn() as any,
  ...overrides,
})

describe('AnalyzeTab', () => {
  it('renders drop zone with upload icon and prompt text', () => {
    render(React.createElement(AnalyzeTab, { vs: mockVs(), onSendText: vi.fn() }))
    expect(screen.getByText('Drop an image here or browse')).toBeInTheDocument()
    expect(screen.getByTestId('icon-upload')).toBeInTheDocument()
  })

  it('shows "Change image" when previewUrl exists', () => {
    render(React.createElement(AnalyzeTab, { vs: mockVs({ previewUrl: 'data:image/png;base64,abc' }), onSendText: vi.fn() }))
    expect(screen.getByText('Change image')).toBeInTheDocument()
  })

  it('renders hidden file input', () => {
    render(React.createElement(AnalyzeTab, { vs: mockVs(), onSendText: vi.fn() }))
    const input = screen.getByLabelText('Upload image for analysis')
    expect(input).toHaveAttribute('type', 'file')
    expect(input).toHaveAttribute('accept', 'image/*')
  })

  it('highlights border on drag over', () => {
    const vs = mockVs({ dragOver: true })
    const { container } = render(React.createElement(AnalyzeTab, { vs, onSendText: vi.fn() }))
    const dropZone = container.querySelector('.border-primary')
    expect(dropZone).toBeInTheDocument()
  })

  it('reverts border on drag leave', () => {
    const vs = mockVs({ dragOver: false })
    const { container } = render(React.createElement(AnalyzeTab, { vs, onSendText: vi.fn() }))
    const dropZone = container.querySelector('.border-border\\/50')
    expect(dropZone).toBeInTheDocument()
  })

  it('displays preview image when previewUrl set', () => {
    render(React.createElement(AnalyzeTab, { vs: mockVs({ previewUrl: 'data:image/png;base64,abc' }), onSendText: vi.fn() }))
    expect(screen.getByAltText('Preview')).toHaveAttribute('src', 'data:image/png;base64,abc')
  })

  it('shows clear button overlay on preview', () => {
    render(React.createElement(AnalyzeTab, { vs: mockVs({ previewUrl: 'data:image/png;base64,abc' }), onSendText: vi.fn() }))
    expect(screen.getByLabelText('Clear preview')).toBeInTheDocument()
  })

  it('renders analysis results section', () => {
    const vs = mockVs({
      analyzeResult: { caption: 'A cat', confidence: 0.95, images_learned: 10, mean_accuracy: 85.5, tags: ['animal'] },
    })
    render(React.createElement(AnalyzeTab, { vs, onSendText: vi.fn() }))
    expect(screen.getByText('A cat')).toBeInTheDocument()
    expect(screen.getByText('0.95')).toBeInTheDocument()
    expect(screen.getByText('animal')).toBeInTheDocument()
  })

  it('shows loading skeleton during analysis', () => {
    render(React.createElement(AnalyzeTab, { vs: mockVs({ analyzeLoading: true }), onSendText: vi.fn() }))
    expect(screen.getAllByTestId('skeleton')).toHaveLength(3)
  })

  it('handles send text action via onSendText callback', () => {
    const onSendText = vi.fn()
    const vs = mockVs({
      analyzeResult: { caption: 'A dog', confidence: 0.9, images_learned: 5, mean_accuracy: 80, tags: [] },
    })
    render(React.createElement(AnalyzeTab, { vs, onSendText }))
    fireEvent.click(screen.getByText('Send caption to chat'))
    expect(onSendText).toHaveBeenCalledWith('A dog')
  })
})
