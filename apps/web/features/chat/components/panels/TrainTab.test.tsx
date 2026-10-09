import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { TrainTab } from './TrainTab'
import type { UseVisionStudioReturn } from './useVisionStudio'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, onClick, disabled, variant, size, className, ...rest }: any) =>
    React.createElement('button', { onClick, disabled, className, ...rest }, children),
  IconUpload: (p: any) => React.createElement('svg', { 'data-testid': 'icon-upload', ...p }),
  IconX: (p: any) => React.createElement('svg', { 'data-testid': 'icon-x', ...p }),

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

const mockVs = (overrides: Record<string, any> = {}): UseVisionStudioReturn => ({
  fileInputRef: { current: null } as any,
  previewUrl: '',
  previewFileName: null,
  clearPreview: vi.fn() as any,
  handleFileSelect: vi.fn() as any,
  trainLabel: '',
  setTrainLabel: vi.fn() as any,
  trainLoading: false,
  handleTrainWithLabel: vi.fn() as any,
  trainResult: null,
  tab: 'train',
  setTab: vi.fn() as any,
  analyzeLoading: false,
  analyzeResult: null,
  analyzeError: '',
  genPrompt: '',
  setGenPrompt: vi.fn() as any,
  genLoading: false,
  genResult: null,
  genError: '',
  setGenResult: vi.fn() as any,
  trainingReport: null,
  resetLoading: false,
  dragOver: false,
  retryLoading: false,
  dropRef: { current: null } as any,
  refreshReport: vi.fn() as any,
  processFile: vi.fn() as any,
  retryAnalyze: vi.fn() as any,
  handleDrop: vi.fn() as any,
  handleGenerateImage: vi.fn() as any,
  handleSendGeneratedImage: vi.fn() as any,
  handleReset: vi.fn() as any,
  setDragOver: vi.fn() as any,
  ...overrides,
})

describe('TrainTab', () => {
  it('renders description text', () => {
    render(React.createElement(TrainTab, { vs: mockVs() }))
    expect(screen.getByText(/Train the vision model/)).toBeInTheDocument()
  })

  it('shows drop zone for image upload', () => {
    render(React.createElement(TrainTab, { vs: mockVs() }))
    expect(screen.getByText('Select an image for training')).toBeInTheDocument()
    expect(screen.getByTestId('icon-upload')).toBeInTheDocument()
  })

  it('displays preview image when previewUrl set', () => {
    render(React.createElement(TrainTab, { vs: mockVs({ previewUrl: 'data:image/png;base64,abc' }) }))
    expect(screen.getByAltText('Training image')).toHaveAttribute('src', 'data:image/png;base64,abc')
  })

  it('shows clear button on preview', () => {
    render(React.createElement(TrainTab, { vs: mockVs({ previewUrl: 'data:image/png;base64,abc' }) }))
    expect(screen.getByLabelText('Remove')).toBeInTheDocument()
  })

  it('renders ground truth caption input', () => {
    render(React.createElement(TrainTab, { vs: mockVs() }))
    expect(screen.getByLabelText('Ground truth label for training image')).toBeInTheDocument()
  })

  it('disables train button when no image or caption', () => {
    render(React.createElement(TrainTab, { vs: mockVs({ previewUrl: '', trainLabel: '' }) }))
    expect(screen.getByText('Train with label')).toBeDisabled()
  })

  it('enables train button when both present', () => {
    render(React.createElement(TrainTab, { vs: mockVs({ previewUrl: 'data:image/png;base64,abc', trainLabel: 'a red car' }) }))
    expect(screen.getByText('Train with label')).not.toBeDisabled()
  })

  it('shows loading state during training', () => {
    render(React.createElement(TrainTab, { vs: mockVs({ previewUrl: 'data:image/png;base64,abc', trainLabel: 'cat', trainLoading: true }) }))
    expect(screen.getByText('Training...')).toBeInTheDocument()
  })

  it('displays training progress', () => {
    render(
      React.createElement(TrainTab, { vs: mockVs({ trainResult: { accuracy: 85.5, caption: 'a red car' } }) })
    )
    expect(screen.getByText('85.5%')).toBeInTheDocument()
    expect(screen.getByText('a red car')).toBeInTheDocument()
  })

  it('calls handleTrainWithLabel when button clicked', () => {
    const vs = mockVs({ previewUrl: 'data:image/png;base64,abc', trainLabel: 'a cat' })
    render(React.createElement(TrainTab, { vs }))
    fireEvent.click(screen.getByText('Train with label'))
    expect(vs.handleTrainWithLabel).toHaveBeenCalled()
  })
})
