// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { datasetLabel, DatasetSelector } from './DatasetSelector'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import { ds, makeDatasets } from './__test-helper'

vi.mock('@/components/DatasetImportDialog', () => ({
  DatasetImportDialog: () => <div data-testid="import-modal" />,
}))

vi.mock('@sloughgpt/strui', () => ({
  Select: ({ children, ...props }: any) => <select data-testid="select" {...props}>{children}</select>,
  SelectTrigger: ({ children }: any) => <div>{children}</div>,
  SelectValue: () => <span>Select a dataset...</span>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,
  Button: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
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

const datasets: UseTrainingDatasetsReturn = makeDatasets()

describe('datasetLabel', () => {
  it('returns name only for basic dataset', () => {
    expect(datasetLabel(ds({ name: 'shakespeare' }))).toContain('shakespeare')
  })

  it('includes sample count', () => {
    expect(datasetLabel(ds({ name: 'ds', samples: 100 }))).toContain('100 samples')
  })

  it('includes source when present', () => {
    expect(datasetLabel(ds({ name: 'ds', source: 'github' }))).toContain('github')
  })

  it('includes size when present', () => {
    expect(datasetLabel(ds({ name: 'ds', size: 2048 }))).toContain('2.0 KB')
  })
})

describe('DatasetSelector', () => {
  afterEach(cleanup)

  it('shows empty state when no datasets', () => {
    render(<DatasetSelector datasets={datasets} value="" onChange={vi.fn()} />)
    expect(screen.getByText(/No datasets/)).toBeDefined()
  })

  it('shows import button in empty state', () => {
    render(<DatasetSelector datasets={datasets} value="" onChange={vi.fn()} />)
    expect(screen.getByText(/Import/)).toBeDefined()
  })

  it('shows selector when datasets exist', () => {
    const withData = { ...datasets, datasets: [ds({ name: 'shakespeare' })] }
    render(<DatasetSelector datasets={withData} value="" onChange={vi.fn()} />)
    expect(screen.getByTestId('select')).toBeDefined()
  })

  it('shows import button when showImport is true', () => {
    const withData = { ...datasets, datasets: [ds({ name: 'shakespeare' })] }
    render(<DatasetSelector datasets={withData} value="" onChange={vi.fn()} showImport />)
    expect(screen.getByText(/Import/)).toBeDefined()
  })
})
