// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
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

import { ApiKeyCreateCard } from './ApiKeyCreateCard'

afterEach(() => cleanup())

describe('ApiKeyCreateCard', () => {
  it('renders create form', () => {
    render(<ApiKeyCreateCard />)
    expect(screen.getByText('Create API Key')).toBeTruthy()
    expect(screen.getByPlaceholderText('Key name')).toBeTruthy()
    expect(screen.getByText('Create')).toBeTruthy()
  })

  it('disables create when empty', () => {
    render(<ApiKeyCreateCard />)
    expect(screen.getByText('Create').hasAttribute('disabled')).toBe(true)
  })

  it('enables create when name entered', () => {
    render(<ApiKeyCreateCard />)
    fireEvent.change(screen.getByPlaceholderText('Key name'), { target: { value: 'my-key' } })
    expect(screen.getByText('Create').hasAttribute('disabled')).toBe(false)
  })

  it('calls onCreate', async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined)
    render(<ApiKeyCreateCard onCreate={onCreate} />)
    fireEvent.change(screen.getByPlaceholderText('Key name'), { target: { value: 'test-key' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => {
      expect(onCreate).toHaveBeenCalledWith('test-key')
    })
  })

  it('clears name after create', async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined)
    render(<ApiKeyCreateCard onCreate={onCreate} />)
    fireEvent.change(screen.getByPlaceholderText('Key name'), { target: { value: 'test' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Key name')).toHaveValue('')
    })
  })
})
