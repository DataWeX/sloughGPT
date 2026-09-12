import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockSend = vi.fn()
const mockStream = vi.fn()
const mockAdd = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/chat-controller', () => ({
  chatController: {
    send: (...args: any[]) => mockSend(...args),
    stream: (...args: any[]) => mockStream(...args),
  },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: { add: (...args: any[]) => mockAdd(...args) },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: any, fallback: string) => e?.message || fallback,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className, ...rest }: any) => <div className={className} {...rest}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    Button: ({ children, className, ...rest }: any) => <button className={className} {...rest}>{children}</button>,
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
}
})

import { QuickActions } from './QuickActions'

function QuickActionsWrapper(props: Partial<React.ComponentProps<typeof QuickActions>> = {}) {
  const [testRunning, setTestRunning] = React.useState(props.testRunning ?? false)
  const [testResponse, setTestResponse] = React.useState<string | null>(props.testResponse ?? null)
  const [knowledgeCount, setKnowledgeCount] = React.useState(props.knowledgeCount ?? 0)
  return (
    <QuickActions
      loading={props.loading ?? false}
      modelStatus={props.modelStatus ?? { loaded: true, model: 'gpt2' }}
      testRunning={testRunning}
      testResponse={testResponse}
      setTestRunning={setTestRunning}
      setTestResponse={setTestResponse}
      knowledgeCount={knowledgeCount}
      setKnowledgeCount={setKnowledgeCount}
    />
  )
}

describe('QuickActions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders nothing when model not loaded', () => {
    const { container } = render(<QuickActionsWrapper modelStatus={{ loaded: false, model: null }} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows Quick test and Quick note cards', () => {
    const { container } = render(<QuickActionsWrapper />)
    expect(container.textContent).toContain('Quick test')
    expect(container.textContent).toContain('Quick note')
  })

  it('calls chatController.stream when Test model clicked', async () => {
    mockStream.mockImplementation(async function* () { yield 'Hello back!' })
    const { container } = render(<QuickActionsWrapper />)
    const buttons = container.querySelectorAll('button')
    const testBtn = Array.from(buttons).find(b => b.textContent?.includes('Test model'))
    fireEvent.click(testBtn!)
    await waitFor(() => {
      expect(mockStream).toHaveBeenCalledWith('Hello!', { waitForModel: true })
    })
  })

  it('shows test response after model responds', async () => {
    mockStream.mockImplementation(async function* () { yield 'Hello back!' })
    const { container } = render(<QuickActionsWrapper />)
    const buttons = container.querySelectorAll('button')
    const testBtn = Array.from(buttons).find(b => b.textContent?.includes('Test model'))
    fireEvent.click(testBtn!)
    await waitFor(() => {
      expect(container.textContent).toContain('Hello back!')
    })
  })

  it('shows error when model call fails', async () => {
    mockStream.mockImplementation(async function* () { throw new Error('Connection refused') })
    const { container } = render(<QuickActionsWrapper />)
    const buttons = container.querySelectorAll('button')
    const testBtn = Array.from(buttons).find(b => b.textContent?.includes('Test model'))
    fireEvent.click(testBtn!)
    await waitFor(() => {
      expect(container.textContent).toContain('Connection refused')
    })
  })

  it('disables button while testing', async () => {
    let resolveTest: any
    mockStream.mockImplementation(async function* () {
      await new Promise(r => { resolveTest = r })
      yield 'done'
    })
    const { container } = render(<QuickActionsWrapper />)
    const buttons = container.querySelectorAll('button')
    const testBtn = Array.from(buttons).find(b => b.textContent?.includes('Test model'))
    fireEvent.click(testBtn!)
    await waitFor(() => {
      expect(container.textContent).toContain('Testing...')
    })
    resolveTest()
  })

  it('saves knowledge note on form submit', async () => {
    mockAdd.mockResolvedValue(undefined)
    const { container } = render(<QuickActionsWrapper />)
    const input = container.querySelector('input[aria-label="Quick add knowledge"]') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'I like TypeScript' } })
    fireEvent.submit(input.closest('form')!)
    await waitFor(() => {
      expect(mockAdd).toHaveBeenCalledWith('I like TypeScript', 'general')
    })
  })

  it('increments knowledgeCount after saving', async () => {
    mockAdd.mockResolvedValue(undefined)
    const { container } = render(<QuickActionsWrapper />)
    const input = container.querySelector('input[aria-label="Quick add knowledge"]') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'fact' } })
    fireEvent.submit(input.closest('form')!)
    await waitFor(() => {
      expect(mockAdd).toHaveBeenCalled()
    })
  })

  it('does not save empty note', async () => {
    const { container } = render(<QuickActionsWrapper />)
    const input = container.querySelector('input[aria-label="Quick add knowledge"]') as HTMLInputElement
    fireEvent.submit(input.closest('form')!)
    expect(mockAdd).not.toHaveBeenCalled()
  })
})
