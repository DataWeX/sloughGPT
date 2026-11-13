// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/model-controller', () => ({
  modelController: { getExportFormats: vi.fn() },
}))
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    exportTrainingPairs: vi.fn(),
    downloadCheckpoint: vi.fn(),
  },
}))
vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))
vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
  downloadBlob: vi.fn(),
}))
vi.mock('@/components/export/ExportHistoryCard', () => ({
  ExportHistoryCard: () => <div data-testid="export-history" />,
  recordExport: vi.fn(),
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  ActionCard: ({ title, actions, children, ...p }: any) => <div data-testid="action-card" {...p}>{title}{actions}{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Badge: ({ children }: any) => <span>{children}</span>,
  IconDownload: () => <span>↓</span>,
  IconRefresh: () => <span>↻</span>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
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

import ExportContent from './ExportContent'
import { modelController } from '@/lib/model-controller'
import { apiGet } from '@/lib/http-client'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(modelController.getExportFormats).mockResolvedValue([
    { key: 'sou', label: 'Soul (.soul)' },
    { key: 'gguf', label: 'GGUF' },
  ] as any)
  vi.mocked(apiGet).mockResolvedValue({ checkpoints: [{ name: 'cp1' }, { name: 'cp2' }] })
})

afterEach(() => cleanup())

describe('ExportContent', () => {
  it('renders section titles', async () => {
    render(<ExportContent />)
    expect(screen.getByText('Model Export')).toBeDefined()
    expect(screen.getByText('Training Data Export')).toBeDefined()
  })

  it('shows export history card', async () => {
    render(<ExportContent />)
    expect(screen.getByTestId('export-history')).toBeDefined()
  })

  it('calls getExportFormats on mount', async () => {
    render(<ExportContent />)
    await waitFor(() => {
      expect(modelController.getExportFormats).toHaveBeenCalled()
    })
  })
})
