// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

afterEach(() => { cleanup() })

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Dialog: ({ children, open, ...props }: any) => open ? <div data-testid="dialog" {...props}>{children}</div> : null,
  DialogContent: ({ children, ...props }: any) => <div data-testid="dialog-content" {...props}>{children}</div>,
  DialogHeader: ({ children, ...props }: any) => <div data-testid="dialog-header" {...props}>{children}</div>,
  DialogTitle: ({ children, ...props }: any) => <span data-testid="dialog-title" {...props}>{children}</span>,
  DialogDescription: ({ children, ...props }: any) => <span data-testid="dialog-desc" {...props}>{children}</span>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
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

vi.mock('@/lib/time-format', () => ({
  formatShortDate: (d: string) => 'Jan 1, 2025',
}))

import { SoulDetailDialog } from './SoulDetailDialog'
import type { Soul } from '@/lib/souls-controller'

const soul: Soul = {
  name: 'test-soul',
  description: 'A detailed soul',
  traits: ['friendly', 'helpful', 'wise'],
  personality: { warmth: 0.9, creativity: 0.7, curiosity: 0.8 },
  cognition: { pattern_recognition: 0.6 },
  emotion: { empathy_depth: 0.85 },
  behavior: { patience: 0.7, confidence: 0.6 },
  generation_params: { temperature: 0.7, top_p: 0.9 },
  lineage: 'gpt-4',
  version: '1.0',
  born_at: '2025-01-01T00:00:00Z',
  size_mb: 4.2,
  epochs_trained: 3,
  final_train_loss: 0.1234,
  final_val_loss: 0.5678,
  training_dataset: '/data/models/test/train.jsonl',
  path: '/models/test-soul.soul',
}

describe('SoulDetailDialog', () => {
  it('renders nothing when soul is null', () => {
    const { container } = render(<SoulDetailDialog soul={null} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(container.querySelector('[data-testid="dialog"]')).toBeNull()
  })

  it('renders dialog when soul provided', () => {
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByTestId('dialog').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('test-soul').length).toBeGreaterThanOrEqual(1)
  })

  it('shows version', () => {
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByText('v1.0').length).toBeGreaterThanOrEqual(1)
  })

  it('shows description', () => {
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByText('A detailed soul').length).toBeGreaterThanOrEqual(1)
  })

  it('shows personality traits', () => {
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Creativity').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Curiosity').length).toBeGreaterThanOrEqual(1)
  })

  it('shows traits badges', () => {
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByText('friendly').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('helpful').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('wise').length).toBeGreaterThanOrEqual(1)
  })

  it('shows cognition section', () => {
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByText('Cognition').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Pattern Recognition').length).toBeGreaterThanOrEqual(1)
  })

  it('shows emotion section', () => {
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByText('Emotion').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Empathy Depth').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onClose when Close clicked', () => {
    const onClose = vi.fn()
    render(<SoulDetailDialog soul={soul} currentSoul={null} onClose={onClose} onSwitch={vi.fn()} />)
    fireEvent.click(screen.getAllByText('Close')[0])
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('shows Switch button when not active soul', () => {
    render(<SoulDetailDialog soul={soul} currentSoul="other-soul" onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.getAllByText(/Switch to/).length).toBeGreaterThanOrEqual(1)
  })

  it('hides Switch button when active soul', () => {
    render(<SoulDetailDialog soul={soul} currentSoul="test-soul" onClose={vi.fn()} onSwitch={vi.fn()} />)
    expect(screen.queryByText(/Switch to test-soul/)).toBeNull()
  })
})
