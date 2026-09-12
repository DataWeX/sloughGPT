// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardFooter: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,

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

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
  importFile: vi.fn(),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

import { SettingsBackupRestoreCard } from './SettingsBackupRestoreCard'

afterEach(() => cleanup())

describe('SettingsBackupRestoreCard', () => {
  it('renders title and description', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    expect(screen.getByText('Backup & restore')).toBeTruthy()
    expect(screen.getByText('Export your settings to a file, or import from a backup')).toBeTruthy()
  })

  it('renders export button', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    expect(screen.getByText('Export settings')).toBeTruthy()
  })

  it('renders import button', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    expect(screen.getByText('Import settings')).toBeTruthy()
  })

  it('export button is not disabled', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    const btn = screen.getByText('Export settings')
    expect(btn.hasAttribute('disabled')).toBe(false)
  })

  it('import button is not disabled', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    const btn = screen.getByText('Import settings')
    expect(btn.hasAttribute('disabled')).toBe(false)
  })

  it('renders version badge when provided', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} version="3.0.0" />)
    expect(screen.getByText(/v3\.0\.0/)).toBeTruthy()
  })
})
