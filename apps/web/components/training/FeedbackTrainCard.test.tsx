// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    trainFromFeedback: vi.fn(),
    get: vi.fn(),
    stop: vi.fn(),
  },
}))
vi.mock('@/lib/souls-controller', () => ({
  soulsController: { loadCheckpoint: vi.fn() },
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Input: (p: any) => <input {...p} />,
  Label: ({ children }: any) => <label>{children}</label>,
  Progress: (p: any) => <div data-testid="progress" data-value={p.value} />,
  AlertDialog: ({ open, children }: any) => open ? <div role="dialog">{children}</div> : null,
  AlertDialogContent: ({ children }: any) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
  AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
  AlertDialogCancel: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,

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

import { FeedbackTrainCard } from './FeedbackTrainCard'
import { trainingJobsController } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'

const mockToast = vi.fn()

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(trainingJobsController.trainFromFeedback).mockResolvedValue({ job_id: 'j1', samples: 10 } as any)
  vi.mocked(trainingJobsController.get).mockResolvedValue(null as any)
  vi.mocked(trainingJobsController.stop).mockResolvedValue(undefined as any)
  vi.mocked(soulsController.loadCheckpoint).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('FeedbackTrainCard', () => {
  it('renders with title and train button', () => {
    render(<FeedbackTrainCard addToast={mockToast} />)
    expect(screen.getAllByText('Train from feedback').length).toBeGreaterThanOrEqual(1)
  })

  it('toggles config panel', () => {
    render(<FeedbackTrainCard addToast={mockToast} />)
    fireEvent.click(screen.getByText('Show config'))
    expect(screen.getByText('Epochs')).toBeDefined()
    expect(screen.getByText('Learning Rate')).toBeDefined()
    expect(screen.getByText('Batch Size')).toBeDefined()
  })

  it('starts training on button click', async () => {
    render(<FeedbackTrainCard addToast={mockToast} />)
    const btns = screen.getAllByText('Train from feedback')
    fireEvent.click(btns[btns.length - 1])
    await waitFor(() => {
      expect(trainingJobsController.trainFromFeedback).toHaveBeenCalled()
    })
  })
})
