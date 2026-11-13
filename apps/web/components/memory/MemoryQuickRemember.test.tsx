// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
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

import { MemoryQuickRemember } from './MemoryQuickRemember'

afterEach(() => { cleanup() })

describe('MemoryQuickRemember', () => {
  it('renders content input', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByPlaceholderText('Quick remember: type a fact and press Enter')).toBeDefined()
  })

  it('renders topic input', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByPlaceholderText('Topic (optional)')).toBeDefined()
  })

  it('renders Remember button', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByText('Remember')).toBeDefined()
  })

  it('disables Remember button when content is empty', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByText('Remember').hasAttribute('disabled')).toBe(true)
  })

  it('shows saving state', () => {
    render(<MemoryQuickRemember content="test" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} remembering />)
    expect(screen.getByText('Saving...')).toBeDefined()
  })

  it('calls onRemember when Remember button is clicked', () => {
    const onRemember = vi.fn()
    render(<MemoryQuickRemember content="fact" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={onRemember} />)
    screen.getByText('Remember').click()
    expect(onRemember).toHaveBeenCalled()
  })
})
