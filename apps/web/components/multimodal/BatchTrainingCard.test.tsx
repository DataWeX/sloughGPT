import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<typeof import('@sloughgpt/strui')>('@sloughgpt/strui')
  return {
    ...actual,
    ProgressBar: ({ value, max, variant }: any) => (
      <div data-testid="progress-bar" data-value={value} data-max={max} data-variant={variant} />
    ),
  
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
}
})

import BatchTrainingCard from './BatchTrainingCard'

const mockOnFileUpload = vi.fn()
const mockOnDirUpload = vi.fn()

describe('BatchTrainingCard', () => {
  afterEach(cleanup)

  it('renders card title', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByText('Train with multiple images')).toBeDefined()
  })

  it('shows description text', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByText(/Train on multiple images at once/)).toBeDefined()
  })

  it('renders Upload images button', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByText('Upload images')).toBeDefined()
  })

  it('shows Starting... when batchUploading is true', () => {
    render(<BatchTrainingCard batchUploading={true} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByText('Starting…')).toBeDefined()
  })

  it('disables Upload button when batchUploading', () => {
    render(<BatchTrainingCard batchUploading={true} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    const btn = screen.getByText('Starting…').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('renders server directory input', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByPlaceholderText('/path/to/images')).toBeDefined()
  })

  it('Train from directory button disabled when input is empty', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    const btn = screen.getByText('Train from directory').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('enables Train from directory when path entered', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    fireEvent.change(screen.getByPlaceholderText('/path/to/images'), { target: { value: '/data/imgs' } })
    const btn = screen.getByText('Train from directory').closest('button')!
    expect(btn.disabled).toBe(false)
  })

  it('calls onDirUpload with path when Train from directory clicked', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    fireEvent.change(screen.getByPlaceholderText('/path/to/images'), { target: { value: '/data/imgs' } })
    fireEvent.click(screen.getByText('Train from directory'))
    expect(mockOnDirUpload).toHaveBeenCalledWith('/data/imgs')
  })

  it('shows progress bar when trainStatus is running', () => {
    const status = { running: true, total: 10, completed: 5, progress_pct: 50, current_image: 'img1.png', job_id: 'j1', errors: 0, current_caption: '' }
    render(<BatchTrainingCard batchUploading={false} trainStatus={status} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByTestId('progress-bar')).toBeDefined()
    expect(screen.getByText('5/10 images')).toBeDefined()
    expect(screen.getByText('50%')).toBeDefined()
  })

  it('shows current image name during training', () => {
    const status = { running: true, total: 10, completed: 3, progress_pct: 30, current_image: 'photo.jpg', job_id: 'j1', errors: 0, current_caption: '' }
    render(<BatchTrainingCard batchUploading={false} trainStatus={status} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByText('Processing: photo.jpg')).toBeDefined()
  })

  it('shows Training... on Train from directory button when training running', () => {
    const status = { running: true, total: 10, completed: 2, progress_pct: 20, current_image: 'a.png', job_id: 'j1', errors: 0, current_caption: '' }
    render(<BatchTrainingCard batchUploading={false} trainStatus={status} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.getByText('Training…')).toBeDefined()
  })

  it('disables Train from directory when training running', () => {
    const status = { running: true, total: 10, completed: 2, progress_pct: 20, current_image: 'a.png', job_id: 'j1', errors: 0, current_caption: '' }
    render(<BatchTrainingCard batchUploading={false} trainStatus={status} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    const btn = screen.getByText('Training…').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('does not show progress when trainStatus is null', () => {
    render(<BatchTrainingCard batchUploading={false} trainStatus={null} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.queryByTestId('progress-bar')).toBeNull()
  })

  it('does not show progress when trainStatus is not running', () => {
    const status = { running: false, total: 10, completed: 10, progress_pct: 100, current_image: '', job_id: 'j1', errors: 0, current_caption: '' }
    render(<BatchTrainingCard batchUploading={false} trainStatus={status} onFileUpload={mockOnFileUpload} onDirUpload={mockOnDirUpload} />)
    expect(screen.queryByTestId('progress-bar')).toBeNull()
  })
})
