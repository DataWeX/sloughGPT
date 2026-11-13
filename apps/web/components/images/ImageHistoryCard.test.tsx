// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
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

vi.mock('@/lib/time-ago', () => ({
  timeAgo: (ts: number) => '2h ago',
}))

import { ImageHistoryCard, recordImageGeneration } from './ImageHistoryCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('ImageHistoryCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<ImageHistoryCard />)
    expect(container.firstChild).toBeNull()
  })

  it('shows history entries', async () => {
    recordImageGeneration('A cat in space', 'realistic')
    recordImageGeneration('Sunset over mountains', 'anime')
    render(<ImageHistoryCard />)
    await waitFor(() => {
      expect(screen.getByText('Generation History')).toBeTruthy()
    })
    expect(screen.getByText('A cat in space')).toBeTruthy()
    expect(screen.getByText('Sunset over mountains')).toBeTruthy()
    expect(screen.getByText('(2)')).toBeTruthy()
  })

  it('filters by style', async () => {
    recordImageGeneration('Cat', 'realistic')
    recordImageGeneration('Dog', 'anime')
    render(<ImageHistoryCard />)
    await waitFor(() => { expect(screen.getByText('Cat')).toBeTruthy() })
    const animeBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('anime'))
    if (animeBtn) fireEvent.click(animeBtn)
    expect(screen.queryByText('Cat')).toBeNull()
    expect(screen.getByText('Dog')).toBeTruthy()
  })

  it('calls onReUse', async () => {
    const onReUse = vi.fn()
    recordImageGeneration('Test prompt', 'realistic')
    render(<ImageHistoryCard onReUse={onReUse} />)
    await waitFor(() => { expect(screen.getByText('Test prompt')).toBeTruthy() })
    const reUseBtn = screen.getAllByRole('button').find(b => b.textContent === 'Re-use')
    if (reUseBtn) fireEvent.click(reUseBtn)
    expect(onReUse).toHaveBeenCalledWith('Test prompt', 'realistic')
  })

  it('deletes an entry', async () => {
    recordImageGeneration('To delete', 'realistic')
    render(<ImageHistoryCard />)
    await waitFor(() => { expect(screen.getByText('To delete')).toBeTruthy() })
    const delBtn = screen.getAllByRole('button').find(b => b.textContent === 'Del')
    if (delBtn) fireEvent.click(delBtn)
    await waitFor(() => {
      expect(screen.queryByText('To delete')).toBeNull()
    })
  })

  it('clears all history', async () => {
    recordImageGeneration('One', 'realistic')
    recordImageGeneration('Two', 'realistic')
    render(<ImageHistoryCard />)
    await waitFor(() => { expect(screen.getByText('One')).toBeTruthy() })
    fireEvent.click(screen.getByText('Clear'))
    await waitFor(() => {
      expect(screen.queryByText('One')).toBeNull()
    })
  })
})
