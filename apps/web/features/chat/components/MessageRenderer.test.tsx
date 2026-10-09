import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { MessageRenderer } from './MessageRenderer'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),

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

vi.mock('./messages/Markdown', () => ({
  Markdown: ({ content }: { content: string }) => React.createElement('div', { 'data-testid': 'markdown' }, content),
}))

vi.mock('./messages/ToolCallPanel', () => ({
  ToolCallPanel: ({ events }: { events: any[] }) =>
    React.createElement('div', { 'data-testid': 'tool-call-panel' }, `${events.length} tool calls`),
}))

vi.mock('./StreamingIndicator', () => ({
  StreamingIndicator: ({ status }: { status: string }) =>
    React.createElement('span', { 'data-testid': 'streaming-indicator', 'data-status': status }),
}))

const baseMessage = {
  id: 'msg-1',
  role: 'user' as const,
  content: 'Hello world',
  timestamp: new Date('2026-09-03'),
}

describe('MessageRenderer', () => {
  it('renders user message as plain text', () => {
    render(React.createElement(MessageRenderer, { message: baseMessage }))
    expect(screen.getByText('Hello world')).toBeInTheDocument()
  })

  it('renders assistant message with Markdown component', () => {
    render(
      React.createElement(MessageRenderer, {
        message: { ...baseMessage, role: 'assistant', content: 'Test response' },
      })
    )
    expect(screen.getByTestId('markdown')).toHaveTextContent('Test response')
  })

  it('shows ThinkingBlock when streaming with no content', () => {
    render(
      React.createElement(MessageRenderer, {
        message: { ...baseMessage, role: 'assistant', content: '' },
        isStreaming: true,
      })
    )
    expect(screen.getByTestId('streaming-indicator')).toHaveAttribute('data-status', 'thinking')
  })

  it('shows StreamingCursor during assistant stream', () => {
    render(
      React.createElement(MessageRenderer, {
        message: { ...baseMessage, role: 'assistant', content: 'Partial' },
        isStreaming: true,
      })
    )
    expect(screen.getByText('Partial')).toBeInTheDocument()
    const cursor = document.querySelector('.animate-pulse')
    expect(cursor).toBeInTheDocument()
  })

  it('renders ImageGrid with multiple images', () => {
    const images = [
      { id: '1', name: 'pic1.png', dataUrl: 'data:image/png;base64,abc' },
      { id: '2', name: 'pic2.jpg', dataUrl: 'data:image/jpeg;base64,def' },
    ]
    render(React.createElement(MessageRenderer, { message: { ...baseMessage, images } }))
    expect(screen.getAllByRole('img')).toHaveLength(2)
  })

  it('renders ImageGrid empty when no images', () => {
    const { container } = render(React.createElement(MessageRenderer, { message: { ...baseMessage, images: [] } }))
    expect(container.querySelectorAll('img')).toHaveLength(0)
  })

  it('renders ToolCallPanel when toolCalls present', () => {
    const toolCalls = [{ tool: 'search', status: 'success' as const, args: {} }]
    render(React.createElement(MessageRenderer, { message: { ...baseMessage, toolCalls } }))
    expect(screen.getByTestId('tool-call-panel')).toHaveTextContent('1 tool calls')
  })

  it('renders ToolCalls empty when no toolCalls', () => {
    const { container } = render(React.createElement(MessageRenderer, { message: baseMessage }))
    expect(container.querySelector('[data-testid="tool-call-panel"]')).not.toBeInTheDocument()
  })

  it('shows reasoning in collapsible details element', () => {
    render(
      React.createElement(MessageRenderer, {
        message: { ...baseMessage, role: 'assistant', reasoning: 'Thinking step...' },
      })
    )
    expect(screen.getByText('Reasoning')).toBeInTheDocument()
    expect(screen.getByText('Thinking step...')).toBeInTheDocument()
  })

  it('hides reasoning when not provided', () => {
    const { container } = render(React.createElement(MessageRenderer, { message: { ...baseMessage, role: 'assistant' } }))
    expect(container.querySelector('details')).not.toBeInTheDocument()
  })

  it('highlights search query in user messages', () => {
    render(React.createElement(MessageRenderer, { message: baseMessage, searchQuery: 'world' }))
    const mark = screen.getByText('world')
    expect(mark.tagName).toBe('MARK')
  })

  it('escapes regex special chars in search query', () => {
    render(
      React.createElement(MessageRenderer, {
        message: { ...baseMessage, content: 'Price is $5.00 (tax incl.)' },
        searchQuery: '$5.00',
      })
    )
    expect(screen.getByText('$5.00')).toBeInTheDocument()
  })

  it('shows error indicator with destructive dot when isError=true', () => {
    render(React.createElement(MessageRenderer, { message: { ...baseMessage, isError: true } }))
    expect(screen.getByText('Response interrupted')).toBeInTheDocument()
  })

  it('displays metadata when provided', () => {
    render(
      React.createElement(MessageRenderer, {
        message: {
          ...baseMessage,
          role: 'assistant',
          metadata: { model: 'gpt-4', tokens: 42, latencyMs: 1200 },
        },
      })
    )
    expect(screen.getByText('gpt-4')).toBeInTheDocument()
    expect(screen.getByText('· 42 tokens')).toBeInTheDocument()
    expect(screen.getByText('· 1200ms')).toBeInTheDocument()
  })

  it('applies custom className prop', () => {
    const { container } = render(
      React.createElement(MessageRenderer, { message: baseMessage, className: 'custom-class' })
    )
    expect(container.firstChild).toHaveClass('custom-class')
  })
})
