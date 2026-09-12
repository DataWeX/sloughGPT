// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
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

import { ConversationDetailCard } from './ConversationDetailCard'

afterEach(() => { cleanup() })

describe('ConversationDetailCard', () => {
  it('renders conversation name', () => {
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={[]} />)
    expect(screen.getByText('Test Chat')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={[]} />)
    expect(screen.getByText('No messages in this conversation.')).toBeTruthy()
  })

  it('renders messages', () => {
    const messages = [
      { role: 'user' as const, content: 'Hello' },
      { role: 'assistant' as const, content: 'Hi there!' },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    expect(screen.getByText('Hello')).toBeTruthy()
    expect(screen.getByText('Hi there!')).toBeTruthy()
    expect(screen.getByText(/2 messages/)).toBeTruthy()
  })

  it('shows message count', () => {
    const messages = [
      { role: 'user' as const, content: 'One' },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    expect(screen.getByText(/1 message/)).toBeTruthy()
  })

  it('truncates long messages', () => {
    const longContent = 'A'.repeat(250)
    const messages = [
      { role: 'user' as const, content: longContent },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    expect(screen.getByText(/Show more/)).toBeTruthy()
  })

  it('expands long message on click', () => {
    const longContent = 'A'.repeat(250)
    const messages = [
      { role: 'user' as const, content: longContent },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    fireEvent.click(screen.getByText(/Show more/))
    expect(screen.getByText(/Show less/)).toBeTruthy()
    expect(screen.getByText(longContent)).toBeTruthy()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={[]} onClose={onClose} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalled()
  })

  it('shows truncated conversation id', () => {
    render(<ConversationDetailCard conversationId="abcdef1234567890" conversationName="Test Chat" messages={[]} />)
    expect(screen.getByText(/abcdef12/)).toBeTruthy()
  })
})
