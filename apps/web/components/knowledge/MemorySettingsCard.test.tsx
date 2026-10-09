// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {
    getConfig: vi.fn(),
    setEnabled: vi.fn(),
    consolidate: vi.fn(),
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))
vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: any) => e?.message || 'Unknown error',
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Switch: ({ checked, onCheckedChange, ...p }: any) => <input type="checkbox" checked={checked} onChange={e => onCheckedChange?.(e.target.checked)} {...p} />,
  Skeleton: (p: any) => <div data-testid="skeleton" {...p} />,

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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

import { MemorySettingsCard } from './MemorySettingsCard'
import { memoryController } from '@/lib/memory-controller'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(memoryController.getConfig).mockResolvedValue({
    enabled: true, max_facts: 100, min_chars: 50,
    archive_retention_days: 30, consolidation_threshold: 10,
  } as any)
  vi.mocked(memoryController.setEnabled).mockResolvedValue(undefined as any)
  vi.mocked(memoryController.consolidate).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('MemorySettingsCard', () => {
  it('calls getConfig on mount', async () => {
    render(<MemorySettingsCard />)
    await waitFor(() => {
      expect(memoryController.getConfig).toHaveBeenCalled()
    })
  })

  it('toggles enabled state', async () => {
    render(<MemorySettingsCard />)
    await waitFor(() => {
      expect(screen.getByRole('checkbox')).toBeDefined()
    })
    fireEvent.click(screen.getByRole('checkbox'))
    await waitFor(() => {
      expect(memoryController.setEnabled).toHaveBeenCalledWith(false)
    })
  })

  it('runs consolidation', async () => {
    render(<MemorySettingsCard />)
    await waitFor(() => {
      expect(screen.getByText('Run Consolidation Now')).toBeDefined()
    })
    fireEvent.click(screen.getByText('Run Consolidation Now'))
    await waitFor(() => {
      expect(memoryController.consolidate).toHaveBeenCalled()
    })
  })
})
