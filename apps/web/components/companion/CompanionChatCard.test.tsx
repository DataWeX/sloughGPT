// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeAll } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

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

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => 'just now',
}))

import { CompanionChatCard } from './CompanionChatCard'

afterEach(() => cleanup())

describe('CompanionChatCard', () => {
  it('renders chat interface', () => {
    render(<CompanionChatCard />)
    expect(screen.getByText('Chat')).toBeTruthy()
    expect(screen.getByPlaceholderText('Say something...')).toBeTruthy()
    expect(screen.getByText('Send')).toBeTruthy()
    expect(screen.getByText('Start a conversation with your companion.')).toBeTruthy()
  })

  it('disables send when empty', () => {
    render(<CompanionChatCard />)
    const sendBtn = screen.getByText('Send')
    expect(sendBtn.hasAttribute('disabled')).toBe(true)
  })

  it('shows user message after sending', async () => {
    const onSend = vi.fn().mockResolvedValue('Hello there!')
    render(<CompanionChatCard onSend={onSend} />)
    const input = screen.getByPlaceholderText('Say something...')
    fireEvent.change(input, { target: { value: 'Hi companion' } })
    fireEvent.click(screen.getByText('Send'))
    await waitFor(() => {
      expect(screen.getByText('Hi companion')).toBeTruthy()
    })
  })

  it('shows assistant response', async () => {
    const onSend = vi.fn().mockResolvedValue('I am doing great!')
    render(<CompanionChatCard onSend={onSend} />)
    fireEvent.change(screen.getByPlaceholderText('Say something...'), { target: { value: 'How are you?' } })
    fireEvent.click(screen.getByText('Send'))
    await waitFor(() => {
      expect(screen.getByText('I am doing great!')).toBeTruthy()
    })
  })

  it('shows error message on failure', async () => {
    const onSend = vi.fn().mockRejectedValue(new Error('Network error'))
    render(<CompanionChatCard onSend={onSend} />)
    fireEvent.change(screen.getByPlaceholderText('Say something...'), { target: { value: 'Test' } })
    fireEvent.click(screen.getByText('Send'))
    await waitFor(() => {
      expect(screen.getByText('Sorry, I encountered an error. Please try again.')).toBeTruthy()
    })
  })

  it('clears messages', async () => {
    const onSend = vi.fn().mockResolvedValue('Response')
    render(<CompanionChatCard onSend={onSend} />)
    fireEvent.change(screen.getByPlaceholderText('Say something...'), { target: { value: 'Hello' } })
    fireEvent.click(screen.getByText('Send'))
    await waitFor(() => { expect(screen.getByText('Hello')).toBeTruthy() })
    fireEvent.click(screen.getByText('Clear'))
    expect(screen.getByText('Start a conversation with your companion.')).toBeTruthy()
  })

  it('sends on Enter key', async () => {
    const onSend = vi.fn().mockResolvedValue('Pong')
    render(<CompanionChatCard onSend={onSend} />)
    const input = screen.getByPlaceholderText('Say something...')
    fireEvent.change(input, { target: { value: 'Ping' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith('Ping')
    })
  })
})
