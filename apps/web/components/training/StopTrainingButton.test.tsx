// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { StopTrainingButton } from './StopTrainingButton'

vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, onClick, disabled, variant, size, className }: any) => (
    <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} data-size={size}>{children}</button>
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
}))

vi.mock('@/components/ConfirmDialog', () => ({
  ConfirmDialog: ({ open, onConfirm, title, description, confirmLabel }: any) => (
    open ? (
      <div data-testid="confirm-dialog">
        <div>{title}</div>
        <div>{description}</div>
        <button onClick={onConfirm}>{confirmLabel}</button>
      </div>
    ) : null
  ),
}))

describe('StopTrainingButton', () => {
  const mockOnStop = vi.fn()
  const mockToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockOnStop.mockResolvedValue(undefined)
  })

  it('renders stop button', () => {
    render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    expect(screen.getByText('Stop training')).toBeTruthy()
  })

  it('opens confirm dialog on click', async () => {
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    expect(screen.getByText('Stop training?')).toBeTruthy()
    unmount()
  })

  it('calls onStop when confirmed', async () => {
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Stop training')[1])
    await waitFor(() => {
      expect(mockOnStop).toHaveBeenCalled()
    })
    expect(mockToast).toHaveBeenCalledWith('Training stopped', 'success')
    unmount()
  })

  it('shows error toast on failure', async () => {
    mockOnStop.mockRejectedValue(new Error('Network error'))
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Stop training')[1])
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not stop training', 'error')
    })
    unmount()
  })

  it('shows stopping state while processing', async () => {
    let resolveFn!: () => void
    mockOnStop.mockImplementation(() => new Promise<void>(r => { resolveFn = r }))
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Stop training')[1])
    await waitFor(() => {
      expect(screen.getAllByText('Stopping...').length).toBeGreaterThan(0)
    })
    resolveFn!()
    await waitFor(() => {
      expect(screen.queryByTestId('confirm-dialog')).toBeNull()
    })
    unmount()
  })
})
