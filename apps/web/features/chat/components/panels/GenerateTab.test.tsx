import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { GenerateTab } from './GenerateTab'
import type { UseVisionStudioReturn } from './useVisionStudio'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, onClick, disabled, variant, size, className, ...rest }: any) =>
    React.createElement('button', { onClick, disabled, className, ...rest }, children),
  IconSend: (p: any) => React.createElement('svg', { 'data-testid': 'icon-send', ...p }),
}))

const mockVs = (overrides: Record<string, any> = {}): UseVisionStudioReturn => ({
  genPrompt: '',
  setGenPrompt: vi.fn() as any,
  genLoading: false,
  genError: '',
  genResult: null,
  handleGenerateImage: vi.fn() as any,
  handleSendGeneratedImage: vi.fn() as any,
  setGenResult: vi.fn() as any,
  tab: 'generate',
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
  trainingReport: null,
  resetLoading: false,
  dragOver: false,
  retryLoading: false,
  fileInputRef: { current: null } as any,
  dropRef: { current: null } as any,
  refreshReport: vi.fn() as any,
  processFile: vi.fn() as any,
  retryAnalyze: vi.fn() as any,
  handleDrop: vi.fn() as any,
  handleFileSelect: vi.fn() as any,
  handleTrainWithLabel: vi.fn() as any,
  handleReset: vi.fn() as any,
  clearPreview: vi.fn() as any,
  setDragOver: vi.fn() as any,
  ...overrides,
})

describe('GenerateTab', () => {
  it('renders prompt input field', () => {
    render(React.createElement(GenerateTab, { vs: mockVs() }))
    expect(screen.getByLabelText('Image generation prompt')).toBeInTheDocument()
  })

  it('renders generate button', () => {
    render(React.createElement(GenerateTab, { vs: mockVs() }))
    expect(screen.getByText('Generate')).toBeInTheDocument()
  })

  it('disables button when prompt empty', () => {
    render(React.createElement(GenerateTab, { vs: mockVs({ genPrompt: '' }) }))
    expect(screen.getByText('Generate')).toBeDisabled()
  })

  it('disables button during loading', () => {
    render(React.createElement(GenerateTab, { vs: mockVs({ genPrompt: 'cat', genLoading: true }) }))
    expect(screen.getByText('Generating...')).toBeDisabled()
  })

  it('calls handleGenerateImage on button click', () => {
    const vs = mockVs({ genPrompt: 'a sunset' })
    render(React.createElement(GenerateTab, { vs }))
    fireEvent.click(screen.getByText('Generate'))
    expect(vs.handleGenerateImage).toHaveBeenCalled()
  })

  it('triggers generation on Enter key', () => {
    const vs = mockVs({ genPrompt: 'a mountain' })
    render(React.createElement(GenerateTab, { vs }))
    fireEvent.keyDown(screen.getByLabelText('Image generation prompt'), { key: 'Enter' })
    expect(vs.handleGenerateImage).toHaveBeenCalled()
  })

  it('shows "Generating..." text during loading', () => {
    render(React.createElement(GenerateTab, { vs: mockVs({ genPrompt: 'cat', genLoading: true }) }))
    expect(screen.getByText('Generating...')).toBeInTheDocument()
  })

  it('displays error message when genError present', () => {
    render(React.createElement(GenerateTab, { vs: mockVs({ genError: 'Rate limited' }) }))
    expect(screen.getByText('Rate limited')).toBeInTheDocument()
  })
})
