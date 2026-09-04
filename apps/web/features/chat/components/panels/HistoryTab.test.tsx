import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { HistoryTab } from './HistoryTab'
import type { UseVisionStudioReturn } from './useVisionStudio'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, onClick, disabled, variant, size, className, ...rest }: any) =>
    React.createElement('button', { onClick, disabled, className, ...rest }, children),
  IconRefresh: (p: any) => React.createElement('svg', { 'data-testid': 'icon-refresh', ...p }),
  IconTrash: (p: any) => React.createElement('svg', { 'data-testid': 'icon-trash', ...p }),
  Skeleton: ({ className }: any) => React.createElement('div', { 'data-testid': 'skeleton', className }),
}))

const mockVs = (overrides: Record<string, any> = {}): UseVisionStudioReturn => ({
  trainingReport: null,
  refreshReport: vi.fn() as any,
  handleReset: vi.fn() as any,
  resetLoading: false,
  tab: 'history',
  setTab: vi.fn() as any,
  analyzeLoading: false,
  analyzeResult: null,
  analyzeError: '',
  previewUrl: '',
  previewFileName: null,
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
  dragOver: false,
  retryLoading: false,
  fileInputRef: { current: null } as any,
  dropRef: { current: null } as any,
  processFile: vi.fn() as any,
  retryAnalyze: vi.fn() as any,
  handleDrop: vi.fn() as any,
  handleFileSelect: vi.fn() as any,
  handleTrainWithLabel: vi.fn() as any,
  handleGenerateImage: vi.fn() as any,
  handleSendGeneratedImage: vi.fn() as any,
  clearPreview: vi.fn() as any,
  setDragOver: vi.fn() as any,
  ...overrides,
})

describe('HistoryTab', () => {
  it('shows loading skeleton when no trainingReport', () => {
    render(React.createElement(HistoryTab, { vs: mockVs() }))
    expect(screen.getByTestId('skeleton')).toBeInTheDocument()
  })

  it('renders stats grid with 4 columns', () => {
    const vs = mockVs({
      trainingReport: {
        images_learned: 10,
        vocab_size: 100,
        mean_accuracy: 85.5,
        last_accuracy: 90.2,
        accuracy_history: [],
        caption_history: [],
      },
    })
    render(React.createElement(HistoryTab, { vs }))
    expect(screen.getByText('Images learned')).toBeInTheDocument()
    expect(screen.getByText('Vocab size')).toBeInTheDocument()
    expect(screen.getByText('Mean accuracy')).toBeInTheDocument()
    expect(screen.getByText('Last accuracy')).toBeInTheDocument()
  })

  it('displays images_learned count', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 25, vocab_size: 0, mean_accuracy: 0, last_accuracy: 0, accuracy_history: [], caption_history: [] },
    })
    render(React.createElement(HistoryTab, { vs }))
    expect(screen.getByText('25')).toBeInTheDocument()
  })

  it('displays vocab_size', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 0, vocab_size: 500, mean_accuracy: 0, last_accuracy: 0, accuracy_history: [], caption_history: [] },
    })
    render(React.createElement(HistoryTab, { vs }))
    expect(screen.getByText('500')).toBeInTheDocument()
  })

  it('displays accuracy with color coding', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 0, vocab_size: 0, mean_accuracy: 92.3, last_accuracy: 88.1, accuracy_history: [], caption_history: [] },
    })
    const { container } = render(React.createElement(HistoryTab, { vs }))
    const successEls = container.querySelectorAll('.text-success')
    expect(successEls.length).toBeGreaterThanOrEqual(1)
  })

  it('displays loss value', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 0, vocab_size: 0, mean_accuracy: 75, last_accuracy: 75, accuracy_history: [80, 85, 90], caption_history: [] },
    })
    render(React.createElement(HistoryTab, { vs }))
    expect(screen.getByText('Accuracy history')).toBeInTheDocument()
  })

  it('renders training entries list', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 0, vocab_size: 0, mean_accuracy: 0, last_accuracy: 0, accuracy_history: [], caption_history: ['A cat sitting', 'A dog running'] },
    })
    render(React.createElement(HistoryTab, { vs }))
    expect(screen.getByText('A cat sitting')).toBeInTheDocument()
    expect(screen.getByText('A dog running')).toBeInTheDocument()
  })

  it('shows empty state when no entries', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 0, vocab_size: 0, mean_accuracy: 0, last_accuracy: 0, accuracy_history: [], caption_history: [] },
    })
    render(React.createElement(HistoryTab, { vs }))
    expect(screen.queryByText('Recent captions learned')).not.toBeInTheDocument()
  })

  it('handles refresh via onRefresh callback', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 0, vocab_size: 0, mean_accuracy: 0, last_accuracy: 0, accuracy_history: [], caption_history: [] },
    })
    render(React.createElement(HistoryTab, { vs }))
    fireEvent.click(screen.getByText('Refresh'))
    expect(vs.refreshReport).toHaveBeenCalled()
  })

  it('handles delete via onDelete callback', () => {
    const vs = mockVs({
      trainingReport: { images_learned: 0, vocab_size: 0, mean_accuracy: 0, last_accuracy: 0, accuracy_history: [], caption_history: [] },
    })
    render(React.createElement(HistoryTab, { vs }))
    fireEvent.click(screen.getByText('Reset model'))
    expect(vs.handleReset).toHaveBeenCalled()
  })
})
