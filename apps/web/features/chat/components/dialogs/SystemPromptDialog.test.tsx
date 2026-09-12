import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const DEFAULTS = [
  { name: 'Helpful Assistant', prompt: 'You are a helpful, harmless, and honest assistant. Answer concisely and accurately.' },
  { name: 'Code Expert', prompt: 'You are an expert software engineer. Provide clean, well-documented code solutions. Explain your reasoning.' },
]

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn(async () => DEFAULTS),
    setKV: vi.fn(async () => {}),
  },
}))

vi.mock('@sloughgpt/strui', () => {
  function Dialog({ children, open }: any) { return open ? <div data-testid="dialog">{children}</div> : null }
  function DialogContent({ children }: any) { return <div data-testid="content">{children}</div> }
  function DialogHeader({ children }: any) { return <div>{children}</div> }
  function DialogTitle({ children }: any) { return <div>{children}</div> }
  function DialogPortal({ children }: any) { return <>{children}</> }
  function DialogOverlay() { return <div data-testid="overlay" /> }
  return {
    Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay,
    Input: ({ value, onChange, onKeyDown, ...props }: any) => (
      <input value={value} onChange={onChange} onKeyDown={onKeyDown} {...props} />
    ),
    Button: ({ children, onClick, ...props }: any) => (
      <button onClick={onClick} {...props}>{children}</button>
    ),
    IconTrash: () => <span data-testid="icon-trash">trash</span>,
    IconPlus: () => <span data-testid="icon-plus">+</span>,
  
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

import { SystemPromptDialog } from './SystemPromptDialog'

describe('SystemPromptDialog', () => {
  afterEach(cleanup)

  it('renders when open', async () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    await waitFor(() => {
      expect(screen.getByTestId('dialog')).toBeDefined()
    })
    expect(screen.getByText('Custom System Prompt')).toBeDefined()
  })

  it('does not render when closed', () => {
    render(<SystemPromptDialog open={false} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    expect(screen.queryByTestId('dialog')).toBeNull()
  })

  it('shows the current value in textarea', async () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="Be helpful" onSave={vi.fn()} />)
    await waitFor(() => {
      const textarea = screen.getByLabelText('System prompt') as HTMLTextAreaElement
      expect(textarea.value).toBe('Be helpful')
    })
  })

  it('shows preset buttons', async () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    await waitFor(() => {
      expect(screen.getByText('Helpful Assistant')).toBeDefined()
      expect(screen.getByText('Code Expert')).toBeDefined()
    })
  })

  it('applies preset text on preset click', async () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    await waitFor(() => {
      expect(screen.getByText('Helpful Assistant')).toBeDefined()
    })
    fireEvent.click(screen.getByText('Helpful Assistant'))
    const textarea = screen.getByLabelText('System prompt') as HTMLTextAreaElement
    expect(textarea.value).toContain('helpful')
  })

  it('calls onSave with draft value when Save changes clicked', async () => {
    const onSave = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={onSave} />)
    await waitFor(() => {
      expect(screen.getByLabelText('System prompt')).toBeDefined()
    })
    const textarea = screen.getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: 'Be concise' } })
    fireEvent.click(screen.getByText('Save changes'))
    expect(onSave).toHaveBeenCalledWith('Be concise')
  })

  it('closes dialog when Save clicked', async () => {
    const onOpenChange = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={onOpenChange} value="" onSave={vi.fn()} />)
    await waitFor(() => {
      expect(screen.getByLabelText('System prompt')).toBeDefined()
    })
    const textarea = screen.getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: 'Be concise' } })
    fireEvent.click(screen.getByText('Save changes'))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('does not save draft when Cancel clicked', async () => {
    const onSave = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={onSave} />)
    await waitFor(() => {
      expect(screen.getByLabelText('System prompt')).toBeDefined()
    })
    const textarea = screen.getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: 'Be concise' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(onSave).not.toHaveBeenCalled()
  })
})
