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
  Input: ({ ...props }: any) => <input {...props} />,

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

import { CollectionCreateCard } from './CollectionCreateCard'

afterEach(() => cleanup())

describe('CollectionCreateCard', () => {
  it('renders create button', () => {
    render(<CollectionCreateCard />)
    expect(screen.getByText('New Pipeline')).toBeTruthy()
    expect(screen.getByText('+ Create')).toBeTruthy()
  })

  it('opens form on click', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    expect(screen.getByPlaceholderText('my-pipeline')).toBeTruthy()
    expect(screen.getByText('Create Pipeline')).toBeTruthy()
    expect(screen.getByText('Cancel')).toBeTruthy()
  })

  it('closes form on cancel', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByPlaceholderText('my-pipeline')).toBeNull()
  })

  it('disables create when empty', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    const createBtn = screen.getByText('Create Pipeline')
    expect(createBtn.hasAttribute('disabled')).toBe(true)
  })

  it('enables create when name entered', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    fireEvent.change(screen.getByPlaceholderText('my-pipeline'), { target: { value: 'test-pipe' } })
    const createBtn = screen.getByText('Create Pipeline')
    expect(createBtn.hasAttribute('disabled')).toBe(false)
  })

  it('calls onCreate with values', async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined)
    render(<CollectionCreateCard onCreate={onCreate} />)
    fireEvent.click(screen.getByText('+ Create'))
    fireEvent.change(screen.getByPlaceholderText('my-pipeline'), { target: { value: 'my-pipe' } })
    fireEvent.click(screen.getByText('Create Pipeline'))
    await waitFor(() => {
      expect(onCreate).toHaveBeenCalledWith('my-pipe', 'file', 'memory')
    })
  })

  it('shows source and store selects', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    const selects = screen.getAllByRole('combobox')
    expect(selects.length).toBe(2)
  })
})
