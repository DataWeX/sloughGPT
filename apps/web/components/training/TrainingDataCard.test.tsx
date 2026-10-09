// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    getTrainingStats: vi.fn(),
    listTrainingPairs: vi.fn(),
    deletePair: vi.fn(),
    updatePairQuality: vi.fn(),
    deleteSyncedPairs: vi.fn(),
  },
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Input: (p: any) => <input {...p} />,

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

import { TrainingDataCard } from './TrainingDataCard'
import { trainingJobsController } from '@/lib/training-controller'

const mockToast = vi.fn()

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(trainingJobsController.getTrainingStats).mockResolvedValue({ total: 100, pending: 10, synced: 50, used: 40 } as any)
  vi.mocked(trainingJobsController.listTrainingPairs).mockResolvedValue({ pairs: [], total: 0 } as any)
  vi.mocked(trainingJobsController.deletePair).mockResolvedValue(undefined as any)
  vi.mocked(trainingJobsController.updatePairQuality).mockResolvedValue(undefined as any)
  vi.mocked(trainingJobsController.deleteSyncedPairs).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('TrainingDataCard', () => {
  it('displays stats', async () => {
    render(<TrainingDataCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('100')).toBeDefined()
      expect(screen.getByText('10')).toBeDefined()
      expect(screen.getByText('50')).toBeDefined()
    })
  })

  it('calls APIs on mount', async () => {
    render(<TrainingDataCard addToast={mockToast} />)
    await waitFor(() => {
      expect(trainingJobsController.listTrainingPairs).toHaveBeenCalled()
      expect(trainingJobsController.getTrainingStats).toHaveBeenCalled()
    })
  })
})
